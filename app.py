"""
app.py — Flask API for the Explainable Hybrid CV System
Run with: python app.py
"""

import os
import sys
import io
import base64
import warnings
import traceback

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
        pil_img   = Image.open(request.files["image"].stream)
        image_rgb = pil_to_rgb224(pil_img)
        image_f   = image_rgb.astype(np.float32) / 255.0
        result    = {"model": model_type, "classes": classes}

        # ── BASELINE CNN path ──────────────────────────────────
        if model_type == "baseline":
            transform    = get_transforms(train=False)
            input_tensor = transform(pil_img.convert("RGB").resize((224, 224)))
            input_tensor = input_tensor.unsqueeze(0).float().to(DEVICE)

            from inference.tta import tta_predict, single_predict
            tta_param = request.args.get("tta", "0")
            
            if tta_param == "1":
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
        return jsonify({"error": traceback.format_exc()}), 500


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Explainable Hybrid CV System — API Server")
    print("  http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)