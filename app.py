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

import cv2
import numpy as np
import torch
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

import matplotlib
matplotlib.use("Agg")

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from skimage.feature import hog as sk_hog, local_binary_pattern

from models.cnn_baseline import CNNBaseline
from data.dataset_loader import get_transforms

app = Flask(__name__)
CORS(app)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

# ── HOG config — MUST match preprocessing/extract_hog_features.py ─
IMG_SIZE     = (128, 128)
ORIENTATIONS = 9
PPC          = (8, 8)
CPB          = (2, 2)
LBP_RADIUS   = 3
LBP_N_POINTS = 8 * LBP_RADIUS   # = 24
COLOR_BINS   = 16


def _hog_of(gray_img):
    return sk_hog(
        gray_img,
        orientations=ORIENTATIONS,
        pixels_per_cell=PPC,
        cells_per_block=CPB,
        block_norm="L2-Hys"
    )


def compute_all_features(image_rgb: np.ndarray) -> np.ndarray:
    """Mirrors extract_hog_features.py exactly."""
    img  = cv2.resize(image_rgb, IMG_SIZE)
    bgr  = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    # 1. HOG spatial pyramid (full image + 2×2 grid)
    hog_parts = [_hog_of(gray)]
    h, w = gray.shape
    for r in range(2):
        for c in range(2):
            patch = gray[r*h//2:(r+1)*h//2, c*w//2:(c+1)*w//2]
            patch = cv2.resize(patch, (64, 64))
            hog_parts.append(_hog_of(patch))
    hog_feat = np.concatenate(hog_parts)

    # 2. Color histogram — HSV, 16 bins per channel → 48-d
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    color_hist = []
    for i, rng in enumerate([(0, 180), (0, 256), (0, 256)]):
        h_hist = cv2.calcHist([hsv], [i], None, [COLOR_BINS], list(rng))
        h_hist = h_hist.flatten().astype(np.float32)
        color_hist.append(h_hist / (h_hist.sum() + 1e-6))
    color_feat = np.concatenate(color_hist)

    # 3. Color moments — mean+std per BGR channel → 6-d
    moments = []
    for ch in range(3):
        c = bgr[:, :, ch].astype(np.float32) / 255.0
        moments += [c.mean(), c.std()]
    moments_feat = np.array(moments, dtype=np.float32)

    # 4. LBP texture → 26-d
    gray_uint8 = (gray * 255).astype(np.uint8)
    lbp = local_binary_pattern(gray_uint8, P=LBP_N_POINTS, R=LBP_RADIUS, method="uniform")
    n_bins = LBP_N_POINTS + 2
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins,
                                range=(0, n_bins), density=True)
    lbp_feat = lbp_hist.astype(np.float32)

    return np.concatenate([hog_feat, color_feat, moments_feat, lbp_feat])


# ── Global model handles ───────────────────────────────────────
baseline_model        = None
baseline_target_layer = None   # Grad-CAM target layer, set to match served arch
hybrid_pipeline       = None   # sklearn Pipeline (Scaler → PCA → LinearSVC)
fusion_model          = None
fusion_cnn            = None
fusion_pca            = None
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
    global baseline_model, hybrid_pipeline, classes, baseline_target_layer
    global fusion_model, fusion_cnn, fusion_pca

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

    # ── Hybrid SVM Pipeline ───────────────────────────────────
    # Tries hybrid_svm.pkl first, falls back to hybrid_svm_pipeline.pkl
    for svm_name in ("hybrid_svm.pkl", "hybrid_svm_pipeline.pkl"):
        svm_path = os.path.join(MODELS_DIR, svm_name)
        if os.path.exists(svm_path):
            try:
                hybrid_pipeline = joblib.load(svm_path)
                print(f"[INFO] Loaded Hybrid SVM pipeline: {svm_name}")
                break
            except Exception as e:
                print(f"[WARN] Could not load {svm_name}: {e}")
                hybrid_pipeline = None

    # ── Fusion Model (Architecture B) ─────────────────────────
    fusion_path = os.path.join(MODELS_DIR, "fusion_best.pth")
    pca_path = os.path.join(ROOT, "data", "hog_pca_model.pkl")
    if os.path.exists(fusion_path) and os.path.exists(pca_path):
        try:
            fusion_pca = joblib.load(pca_path)  # scaler + pca bundle

            # Load CNN + fusion model from checkpoint
            from models.hybrid_fusion import HybridFusion

            ckpt = torch.load(fusion_path, map_location=DEVICE)
            
            fusion_cnn = CNNBaseline(num_classes)
            fusion_cnn.load_state_dict(ckpt['cnn_state'])
            fusion_cnn.to(DEVICE).eval()
            
            fusion_model = HybridFusion(cnn_dim=2048, hog_dim=512, num_classes=num_classes)
            fusion_model.load_state_dict(ckpt['fusion_state'])
            fusion_model.to(DEVICE).eval()
            
            print("[LOADED] Hybrid Fusion model (Architecture B)")
        except Exception as e:
            print(f"[WARNING] Could not load fusion model: {e}")
            fusion_model = None
            fusion_cnn = None
            fusion_pca = None


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


def decision_to_probs(decision_scores: np.ndarray) -> np.ndarray:
    """
    Convert LinearSVC decision_function scores to pseudo-probabilities
    via softmax. LinearSVC has no predict_proba, so this gives a
    calibrated confidence for the frontend confidence display.
    """
    d = decision_scores - decision_scores.max()   # numerical stability
    e = np.exp(d)
    return e / e.sum()


# ── Health endpoint ────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":          "ok",
        "device":          str(DEVICE),
        "baseline_loaded": baseline_model  is not None,
        "hybrid_loaded":   hybrid_pipeline is not None,
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

    if model_type == "baseline" and baseline_model is None:
        return jsonify({"error": "Baseline model not loaded. "
                                 "Run: python training/train_baseline.py"}), 503
    if model_type == "hybrid" and hybrid_pipeline is None:
        return jsonify({"error": "Hybrid SVM model not loaded. "
                                 "Run: python training/train_hybrid_svm.py"}), 503
    if model_type == "fusion":
        if fusion_model is None:
            # Graceful fallback to baseline
            if baseline_model is not None:
                print("[WARN] Fusion model not loaded, falling back to baseline")
                model_type = "baseline"
            else:
                return jsonify({"error": "Fusion model and baseline model not loaded."}), 503
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

        # ── HYBRID SVM path ────────────────────────────────────
        elif model_type == "hybrid":
            hog_feat = compute_all_features(image_rgb).reshape(1, -1)

            # decision_function gives one score per class
            decision = hybrid_pipeline.decision_function(hog_feat)[0]
            probs    = decision_to_probs(decision)

            pred_idx = int(np.argmax(probs))
            top5_idx = np.argsort(probs)[::-1][:5]

            result.update({
                "prediction": classes[pred_idx],
                "confidence": float(probs[pred_idx]),
                "top5": [
                    {"class": classes[i], "prob": float(probs[i])}
                    for i in top5_idx
                ],
                "note": "Hybrid model uses HOG + SVM — Grad-CAM not applicable.",
            })

        # ── FUSION MODEL path ──────────────────────────────────
        elif model_type == "fusion":
            # 1. HOG features -> PCA
            hog_feat = compute_all_features(image_rgb).reshape(1, -1)
            scaler = fusion_pca['scaler']
            pca = fusion_pca['pca']
            hog_scaled = scaler.transform(hog_feat)
            hog_pca = pca.transform(hog_scaled)
            hog_tensor = torch.from_numpy(hog_pca).float().to(DEVICE)
            
            # 2. CNN embedding
            transform = get_transforms(train=False)
            input_tensor = transform(pil_img.convert("RGB").resize((224, 224)))
            input_tensor = input_tensor.unsqueeze(0).float().to(DEVICE)
            
            with torch.no_grad():
                cnn_emb = fusion_cnn.get_embedding(input_tensor)
                logits = fusion_model(cnn_emb, hog_tensor)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                
            pred_idx = int(np.argmax(probs))
            top5_idx = np.argsort(probs)[::-1][:5]
            
            result.update({
                "prediction": classes[pred_idx],
                "confidence": float(probs[pred_idx]),
                "top5": [
                    {"class": classes[i], "prob": float(probs[i])}
                    for i in top5_idx
                ],
                "note": "Fusion model prediction.",
            })

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