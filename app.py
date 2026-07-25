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

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import numpy as np
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

import matplotlib
matplotlib.use("Agg")

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from data.dataset_loader import get_transforms

app = Flask(__name__)
CORS(app)

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


def build_baseline_model(num_classes: int):
    """Return (loaded_model, gradcam_target_layer) for the best available CNN.

    Prefers the Phase-2 ResNet-50 (Places365) at models/phase2_best.pth, loading
    the winning raw/EMA weights recorded in the checkpoint (best_is_ema). Falls
    back to the ResNet-18 baseline.pth (train_baseline.build_model) when Phase 2
    is absent. baseline.pth is a raw ResNet-18 state_dict — loading it into
    CNNBaseline (ResNet-50) was the serving crash (audit N3).
    """
    from utils.checkpoint import load_checkpoint
    phase2 = os.path.join(MODELS_DIR, "phase2_best.pth")
    if os.path.exists(phase2):
        from models.cnn_baseline import CNNBaseline
        meta     = torch.load(phase2, map_location="cpu", weights_only=True)
        backbone = meta.get("backbone", "resnet50_places365_local")
        load_ema = bool(meta.get("best_is_ema", False))
        model = CNNBaseline(num_classes, backbone=backbone)
        load_checkpoint(phase2, model, device=DEVICE, load_ema=load_ema)
        print(f"[INFO] Serving Phase-2 {backbone} (load_ema={load_ema}, "
              f"val_acc={meta.get('val_acc', float('nan')):.2f}%)")
        return model, model.model.layer4[-1]

    from training.train_baseline import build_model
    model = build_model(num_classes)
    load_checkpoint(os.path.join(MODELS_DIR, "baseline.pth"), model, device=DEVICE)
    print("[INFO] Serving ResNet-18 baseline (phase2_best.pth not found)")
    return model, model.layer4[-1]

MODELS_DIR = os.path.join(ROOT, "models")
DATA_DIR   = os.path.join(ROOT, "data", "MIT_Indoor")

# If top-1 confidence is below this, flag prediction as out-of-scope
CONFIDENCE_THRESHOLD = 0.30

# ── Upload validation limits (B-12, light) ─────────────────────
# The public /predict endpoint accepts arbitrary uploads, so bound what we are
# willing to decode: an unbounded or hostile image is a real liability once the
# demo is reachable from the internet.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024        # 10 MB request body
ALLOWED_FORMATS  = {"JPEG", "PNG", "WEBP", "BMP"}
MIN_SIDE_PX      = 32                      # below this the CNN input is meaningless
MAX_SIDE_PX      = 10_000
MAX_TOTAL_PIXELS = 40_000_000              # ~40 MP decompression-bomb guard

# Flask rejects oversized bodies before our handler runs (returns 413).
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
# Pillow raises DecompressionBombError past this instead of allocating the pixels.
Image.MAX_IMAGE_PIXELS = MAX_TOTAL_PIXELS


class UploadError(ValueError):
    """A rejected upload. Carries the HTTP status the caller should receive.

    Separate from unexpected server faults so that bad input yields a clean 4xx
    with a short, user-facing reason instead of a 500.
    """

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message = message
        self.status = status


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


def discover_classes():
    for split in ("train", "test"):
        d = os.path.join(DATA_DIR, split)
        if os.path.isdir(d):
            return sorted(x for x in os.listdir(d)
                          if os.path.isdir(os.path.join(d, x)))
    return []


def load_models():
    global baseline_model, classes, baseline_target_layer

    classes     = discover_classes()
    num_classes = max(len(classes), 1)

    # ── Baseline CNN (Phase-2 ResNet-50 if present, else ResNet-18) ────────
    phase2_path   = os.path.join(MODELS_DIR, "phase2_best.pth")
    baseline_path = os.path.join(MODELS_DIR, "baseline.pth")
    if os.path.exists(phase2_path) or os.path.exists(baseline_path):
        try:
            baseline_model, baseline_target_layer = build_baseline_model(num_classes)
            baseline_model.to(DEVICE).eval()
        except Exception as e:
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
        print(f"       Run: python calibration/run_calibration.py")

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


load_models()


# ── Upload validation ──────────────────────────────────────────
def load_validated_image(file_storage) -> Image.Image:
    """Validate an uploaded file and return a decodable PIL image.

    Checks, in order: non-empty and within the byte limit; decodes as a real
    image of an allowed format (the declared filename/content-type is never
    trusted); and has sane dimensions. Raises UploadError with an appropriate
    4xx status; the caller turns that into a JSON error response.
    """
    stream = file_storage.stream
    stream.seek(0, os.SEEK_END)
    size = stream.tell()
    stream.seek(0)

    if size == 0:
        raise UploadError("Uploaded file is empty.")
    if size > MAX_UPLOAD_BYTES:
        raise UploadError(
            f"Image exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.", 413)

    # Decode-verify: the only trustworthy signal that this is really an image.
    try:
        probe = Image.open(stream)
        fmt = (probe.format or "").upper()
        probe.verify()                      # catches truncated / corrupt payloads
    except Image.DecompressionBombError:
        raise UploadError(
            "Image is too large to decode safely (decompression-bomb guard).", 413) from None
    except UploadError:
        raise
    except Exception:
        raise UploadError("File is not a readable image.") from None

    if fmt not in ALLOWED_FORMATS:
        raise UploadError(
            f"Unsupported image format {fmt or 'unknown'!r}. "
            f"Allowed: {', '.join(sorted(ALLOWED_FORMATS))}.")

    # verify() consumes the file object, so reopen for actual use.
    stream.seek(0)
    img = Image.open(stream)

    w, h = img.size
    if w < MIN_SIDE_PX or h < MIN_SIDE_PX:
        raise UploadError(
            f"Image is too small ({w}x{h}); minimum is {MIN_SIDE_PX}x{MIN_SIDE_PX}.")
    if w > MAX_SIDE_PX or h > MAX_SIDE_PX:
        raise UploadError(
            f"Image is too large ({w}x{h}); maximum side is {MAX_SIDE_PX}px.")
    if w * h > MAX_TOTAL_PIXELS:
        raise UploadError(
            f"Image has too many pixels ({w * h}); maximum is {MAX_TOTAL_PIXELS}.")
    return img


@app.errorhandler(413)
def handle_too_large(_err):
    """Flask aborts oversized bodies before the view runs; keep that JSON."""
    return jsonify({"error": f"Upload exceeds the "
                             f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit."}), 413


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
        return jsonify({"error": "Baseline model not loaded. "
                                 "Run: python training/train_baseline.py"}), 503
    if not classes:
        return jsonify({"error": "No classes found. "
                                 "Check data/MIT_Indoor/train exists."}), 503

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

        result["original_image"] = ndarray_to_b64(image_rgb)
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


@app.route("/predict_batch", methods=["POST"])
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