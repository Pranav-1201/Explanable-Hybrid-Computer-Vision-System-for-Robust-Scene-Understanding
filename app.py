"""
app.py — Flask API for the Explainable Hybrid CV System
Run with: python app.py
"""

import os
import sys
import io
import base64
import warnings
import logging
import uuid
import threading

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import numpy as np
import torch
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image

import matplotlib
matplotlib.use("Agg")

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from data.dataset_loader import get_transforms
from serving.artifacts import ArtifactError, ensure_weights, load_classes, load_manifest
from serving.config import cors_origins, env_flag, predict_rate_limit
from serving.uploads import (MAX_REQUEST_BYTES, UploadError, load_validated_image)

app = Flask(__name__)
CORS(app, origins=cors_origins())

# In-memory storage: fine for the single waitress process this serves today
# (B4). Move to a shared backend (e.g. Redis) before running multiple
# processes/replicas, or the limit becomes per-process instead of global.
limiter = Limiter(get_remote_address, app=app, headers_enabled=True,
                  storage_uri="memory://")


@app.after_request
def _security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    # frontend/index.html is a single self-contained file: no external
    # scripts, styles or fonts, so a strict same-origin policy costs nothing.
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    return response

# Unhandled-error tracebacks are logged, never returned to the client (N14).
# Configure a sink so app.logger.exception() is actually recorded rather than
# dropped - the point of hiding the trace from the caller is that an operator
# can still find it here.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

# Classical HOG/colour/LBP features come from the single source of truth in
# preprocessing/extract_hog_features.extract_features_from_rgb (imported above).
# Serving must NOT duplicate the resize/feature logic — the old duplicate here
# resized RAW->224->128 and drifted from training's RAW->128 (audit N8).


# ── Global model handles ───────────────────────────────────────
baseline_model        = None
baseline_target_layer = None   # Grad-CAM target layer, set to match served arch
classes               = []

# One model, shared by every waitress thread. Grad-CAM registers forward and
# backward hooks on it, and concurrent forward passes would interleave those
# hooks, so all model execution is serialised. Upload parsing and validation
# still run concurrently. On CPU this costs little throughput.
INFERENCE_LOCK = threading.Lock()


MODELS_DIR = os.path.join(ROOT, "models")
CLASSES_PATH = os.path.join(MODELS_DIR, "classes.json")
MANIFEST_PATH = os.path.join(MODELS_DIR, "serving_manifest.json")


def build_baseline_model(num_classes: int):
    """Return (loaded_model, gradcam_target_layer) for the served checkpoint.

    Weights are the EMA-only export described in models/serving_manifest.json.
    They are downloaded and sha256-verified on first start when absent, so a
    fresh clone or container needs neither the dataset nor hand-placed files.
    MODEL_PATH relocates the file (e.g. a container volume); MODEL_URL
    overrides the download source. The checksum is enforced either way.
    """
    from models.cnn_baseline import CNNBaseline
    from utils.checkpoint import load_checkpoint

    manifest = load_manifest(MANIFEST_PATH)
    path = os.environ.get("MODEL_PATH") or os.path.join(MODELS_DIR, manifest["filename"])
    url = os.environ.get("MODEL_URL") or manifest["url"]
    ensure_weights(path, url, manifest["sha256"], manifest["size"])

    meta = torch.load(path, map_location="cpu", weights_only=True)
    if int(meta["num_classes"]) != num_classes:
        raise ArtifactError(f"{path} has {meta['num_classes']} outputs but "
                            f"classes.json lists {num_classes} classes")
    model = CNNBaseline(num_classes, backbone=meta["backbone"], pretrained=False)
    load_checkpoint(path, model, device=DEVICE)
    print(f"[INFO] Serving {meta['backbone']} EMA from {path} "
          f"(val_acc={meta['val_acc']:.2f}%)")
    return model, model.model.layer4[-1]

# If top-1 confidence is below this, flag prediction as out-of-scope
CONFIDENCE_THRESHOLD = 0.30

app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_BYTES


# ── Real-estate scoping (B-15) ─────────────────────────────────
# This stays a GENERAL 67-class MIT Indoor scene model; it was NOT retrained or
# purpose-built for real estate. It is deliberately SCOPED into a focused
# interior/property tagger: a prediction in this curated home-relevant subset is
# surfaced as a room tag, and anything else is routed to the review queue as
# out-of-scope (reusing the existing calibrated rejection machinery, not a new
# model). Keys are raw MIT class dirs; values are display labels for the UI.
HOME_CLASS_LABELS = {
    "artstudio":     "Art studio",
    "bar":           "Bar / lounge",
    "bathroom":      "Bathroom",
    "bedroom":       "Bedroom",
    "children_room": "Children's room",
    "closet":        "Closet",
    "corridor":      "Hallway",
    "dining_room":   "Dining room",
    "gameroom":      "Game room",
    "garage":        "Garage",
    "greenhouse":    "Greenhouse / sunroom",
    "gym":           "Home gym",
    "kitchen":       "Kitchen",
    "laundromat":    "Laundry room",
    "library":       "Home library",
    "livingroom":    "Living room",
    "lobby":         "Lobby / entryway",
    "nursery":       "Nursery",
    "office":        "Home office",
    "pantry":        "Pantry",
    "poolinside":    "Indoor pool",
    "stairscase":    "Staircase",
    "studiomusic":   "Studio",
    "winecellar":    "Wine cellar",
}
MAX_BATCH_IMAGES = 50


def pretty_label(cls: str) -> str:
    """Display label for a raw class: curated home label if present, else a
    title-cased fallback so out-of-scope predictions still read cleanly."""
    return HOME_CLASS_LABELS.get(cls, cls.replace("_", " ").title())


def load_models(strict: bool = False):
    global baseline_model, classes, baseline_target_layer

    try:
        classes = load_classes(CLASSES_PATH)
    except ArtifactError as e:
        if strict:
            raise
        print(f"[WARN] {e}")
        classes = []

    try:
        if not classes:
            raise ArtifactError("no classes loaded; cannot size the model head")
        baseline_model, baseline_target_layer = build_baseline_model(len(classes))
        baseline_model.to(DEVICE).eval()
    except Exception as e:
        if strict:
            raise
        print(f"[WARN] Could not load baseline model: {e}")
        baseline_model = None
        baseline_target_layer = None

    # Load temperature scaler at startup
    global temperature_scaler
    temperature_scaler = None
    TEMP_PATH = os.path.join(MODELS_DIR, 'temperature_cnn.json')
    if os.path.exists(TEMP_PATH):
        from calibration.temperature_scaling import TemperatureScaler
        temperature_scaler = TemperatureScaler.load(TEMP_PATH)
        print(f"[LOADED] Temperature scaler  T={temperature_scaler.T:.4f}")
    else:
        print("[INFO] No temperature scaler found. Confidence scores are uncalibrated.")
        print("       Run: python calibration/run_calibration.py")

    # Rejection threshold from calibrated confidence (B-8). The 0.30 default was
    # fit against UNCALIBRATED confidence and over-rejected valid predictions;
    # calibration/run_calibration.py derives a threshold on calibrated max-prob.
    global CONFIDENCE_THRESHOLD
    CFG_PATH = os.path.join(MODELS_DIR, 'calibration_config.json')
    if os.path.exists(CFG_PATH):
        import json as _json
        with open(CFG_PATH) as _f:
            _cfg = _json.load(_f)
        CONFIDENCE_THRESHOLD = float(_cfg.get('rejection_threshold', CONFIDENCE_THRESHOLD))
        print(f"[LOADED] Calibrated rejection threshold = {CONFIDENCE_THRESHOLD:.4f}")
    else:
        print(f"[INFO] No calibration_config.json; using default threshold {CONFIDENCE_THRESHOLD}")

    # ── Hybrid HOG-SVM arm: RETIRED (B-8) ─────────────────────
    # The HOG->LinearSVC arm scored 10.75% top-1 (67-class, ~7x chance)
    # and is no longer served. Its invalid softmax-over-decision_function
    # confidence (audit N9) is deleted rather than patched. Root cause and
    # the principled replacement are logged in AUDIT_REPORT.md (B-8a).

    # ── Fusion Model (Architecture B): DISABLED from serving (B-8) ─────────
    # HybridFusion (CNN 2048 + HOG-PCA 512) is a real jointly-trained model, but
    # the B-7 ablation showed it does NOT beat CNN-only: fusion 82.01% test vs
    # CNN-only 83.21% (-1.2%). It is intentionally NOT loaded or served -- the
    # app serves the honest best model (CNN-only). The code and checkpoint are
    # retained as a documented research finding for the B-10 ablation and README:
    #   model:      models/hybrid_fusion.py  (HybridFusion, ablate_hog())
    #   training:   training/train_fusion.py (two-stage; prints test top-1)
    #   checkpoint: models/fusion_best.pth


# STRICT_STARTUP=1 (set in the container) turns a missing or unverifiable
# artifact into a startup failure instead of a server that only returns 503s.
load_models(strict=env_flag("STRICT_STARTUP"))


@app.errorhandler(413)
def handle_too_large(_err):
    """Flask aborts oversized bodies before the view runs; keep that JSON."""
    return jsonify({"error": f"Request exceeds the {MAX_REQUEST_BYTES // (1024 * 1024)} MB "
                             f"limit; send fewer photos per request."}), 413


# ── Utilities ──────────────────────────────────────────────────
def pil_to_rgb224(pil_img):
    return np.array(pil_img.convert("RGB").resize((224, 224), Image.LANCZOS))


def ndarray_to_b64(arr):
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def run_gradcam(model, input_tensor, target_layer, pred_idx, image_float):
    cam     = GradCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(pred_idx)]
    gs_cam  = cam(input_tensor=input_tensor.clone().detach(),
                  targets=targets)[0]
    return ndarray_to_b64(
        show_cam_on_image(image_float, gs_cam, use_rgb=True)
    )


# ── Health endpoint ────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":          "ok",
        "device":          str(DEVICE),
        "baseline_loaded": baseline_model  is not None,
        "num_classes":     len(classes),
        "classes":         classes,
    })


@app.route("/classes", methods=["GET"])
def get_classes():
    return jsonify({"classes": classes})


# ── Predict endpoint ───────────────────────────────────────────
@app.route("/predict", methods=["POST"])
@limiter.limit(predict_rate_limit())
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    model_type = request.form.get("model", "baseline")

    # Served arms (B-8): CNN-only. The HOG-SVM arm is retired and the fusion
    # arm is disabled from serving (research finding, -1.2% vs CNN-only).
    SERVED_MODELS = {"baseline"}
    if model_type not in SERVED_MODELS:
        return jsonify({"error": f"Unknown model '{model_type}'. "
                                 f"Served models: {sorted(SERVED_MODELS)}. "
                                 f"Fusion is a disabled research arm "
                                 f"(see README ablation)."}), 400

    if model_type == "baseline" and baseline_model is None:
        return jsonify({"error": "Model not loaded; see the server log."}), 503
    if not classes:
        return jsonify({"error": "No classes loaded. Check models/classes.json."}), 503

    try:
        pil_img = load_validated_image(request.files["image"])
    except UploadError as e:
        return jsonify({"error": e.message}), e.status

    try:
        image_rgb = pil_to_rgb224(pil_img)
        image_f   = image_rgb.astype(np.float32) / 255.0
        result    = {"model": model_type, "classes": classes}

        # ── BASELINE CNN path ──────────────────────────────────
        if model_type == "baseline":
            transform    = get_transforms(train=False)
            input_tensor = transform(pil_img.convert("RGB").resize((224, 224)))
            input_tensor = input_tensor.unsqueeze(0).float().to(DEVICE)

            from inference.tta import tta_predict, single_predict
            # TTA is the served default: measured +0.67 top-1 (83.21 -> 83.88,
            # B-10 ablation) for +~22 ms/image (7 forward passes vs 1), which is
            # imperceptible in the demo. Opt out per-request with tta=0.
            tta_param = request.values.get("tta", "1")

            with INFERENCE_LOCK:
                if tta_param != "0":
                    try:
                        probs = tta_predict(baseline_model, pil_img.convert("RGB"), DEVICE, scaler=temperature_scaler).numpy()
                        result["tta_used"] = True
                    except Exception as e:
                        probs = single_predict(baseline_model, pil_img.convert("RGB"), DEVICE, scaler=temperature_scaler).numpy()
                        result["tta_used"] = False
                        result["tta_error"] = str(e)
                else:
                    probs = single_predict(baseline_model, pil_img.convert("RGB"), DEVICE, scaler=temperature_scaler).numpy()
                    result["tta_used"] = False

                result["calibrated"] = temperature_scaler is not None
                result["temperature"] = temperature_scaler.T if temperature_scaler else 1.0

                pred_idx = int(np.argmax(probs))
                top5_idx = np.argsort(probs)[::-1][:5]

                result.update({
                    "prediction": classes[pred_idx],
                    "confidence": float(probs[pred_idx]),
                    "top5": [
                        {"class": classes[i], "prob": float(probs[i])}
                        for i in top5_idx
                    ],
                })

                try:
                    result["gradcam"] = run_gradcam(
                        baseline_model, input_tensor,
                        baseline_target_layer,
                        pred_idx, image_f
                    )
                except Exception as e:
                    result["gradcam_error"] = str(e)

        # ── Confidence-based rejection ──────────────────────
        if result.get("confidence", 1.0) < CONFIDENCE_THRESHOLD:
            result["original_prediction"] = result["prediction"]
            result["prediction"] = "Unknown / Out of Scope"
            result["out_of_scope"] = True
        else:
            result["out_of_scope"] = False

        return jsonify(result)

    except Exception:
        # audit N14: never hand a traceback to the client - it leaks absolute
        # paths, source structure and dependency versions. Log the full trace
        # server-side against a correlation id and return only that id, so an
        # operator can still tie a user report back to the exact failure.
        error_id = uuid.uuid4().hex[:12]
        app.logger.exception("Unhandled error in /predict (error_id=%s)", error_id)
        return jsonify({
            "error": "Internal error while processing the image.",
            "error_id": error_id,
        }), 500


FRONTEND_DIR = os.path.join(ROOT, "frontend")


@app.route("/")
def index():
    """Serve the interior-tagger single-page app (same-origin as the API)."""
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/predict_batch", methods=["POST"])
@limiter.limit(predict_rate_limit())
def predict_batch():
    """Batch interior tagging for the real-estate workflow (B-15).

    Accepts multiple images (repeatable form field 'images'), runs the served
    CNN on each, and returns a per-image room tag plus a review flag. An image
    is routed to the review queue when the top-1 confidence is below the
    calibrated rejection threshold (low_confidence) or the predicted class is
    outside the curated home subset (out_of_scope). Grad-CAM is intentionally
    omitted here for throughput - the single /predict endpoint provides it on
    demand. Per-file failures are reported inline and never abort the batch.
    """
    if baseline_model is None:
        return jsonify({"error": "Baseline model not loaded."}), 503
    if not classes:
        return jsonify({"error": "No classes found."}), 503

    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No images provided (form field 'images')."}), 400
    if len(files) > MAX_BATCH_IMAGES:
        return jsonify({"error": f"Too many images ({len(files)}); "
                                 f"max {MAX_BATCH_IMAGES} per batch."}), 400

    use_tta = request.values.get("tta", "1") != "0"
    from inference.tta import tta_predict, single_predict

    try:
        results, tagged, review = [], 0, 0
        for fs in files:
            name = fs.filename or "(unnamed)"

            # Per-file validation errors are reported, not fatal to the batch.
            try:
                pil = load_validated_image(fs)
            except UploadError as e:
                results.append({"filename": name, "error": e.message,
                                "review": True, "review_reason": "invalid"})
                review += 1
                continue

            rgb = pil.convert("RGB")
            with INFERENCE_LOCK:
                if use_tta:
                    probs = tta_predict(baseline_model, rgb, DEVICE,
                                        scaler=temperature_scaler).numpy()
                else:
                    probs = single_predict(baseline_model, rgb, DEVICE,
                                           scaler=temperature_scaler).numpy()

            idx = int(np.argmax(probs))
            cls = classes[idx]
            conf = float(probs[idx])
            top5_idx = np.argsort(probs)[::-1][:5]

            low_conf = conf < CONFIDENCE_THRESHOLD
            out_scope = cls not in HOME_CLASS_LABELS
            in_scope = not (low_conf or out_scope)
            reason = "low_confidence" if low_conf else ("out_of_scope" if out_scope else None)
            tagged += int(in_scope)
            review += int(not in_scope)

            results.append({
                "filename":      name,
                "prediction":    cls,
                "label":         pretty_label(cls),
                "confidence":    conf,
                "in_scope":      in_scope,
                "review":        not in_scope,
                "review_reason": reason,
                "top5": [
                    {"class": classes[i], "label": pretty_label(classes[i]),
                     "prob": float(probs[i])}
                    for i in top5_idx
                ],
            })

        return jsonify({
            "results":   results,
            "summary":   {"n": len(files), "tagged": tagged, "review": review},
            "threshold": CONFIDENCE_THRESHOLD,
            "tta_used":  use_tta,
        })

    except Exception:
        error_id = uuid.uuid4().hex[:12]
        app.logger.exception("Unhandled error in /predict_batch (error_id=%s)", error_id)
        return jsonify({"error": "Internal error while processing the batch.",
                        "error_id": error_id}), 500


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Explainable Hybrid CV System — API Server")
    print("  http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)