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

# ── Make sure all project modules are importable ──────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

import matplotlib
matplotlib.use("Agg")          # Non-interactive backend (no display needed)
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from models.cnn_baseline import CNNBaseline
from models.hybrid_cnn import HybridCNN
from classical_features.feature_stack import stack_features
from data.dataset_loader import get_transforms


# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)   # Allow the HTML frontend to call this API

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

# ── Model / class state ───────────────────────────────────────────────────────
baseline_model = None
hybrid_model   = None
classes        = []
hog_feature_dim = None   # Will be set after first stack_features call

MODELS_DIR = os.path.join(ROOT, "models")
DATA_DIR   = os.path.join(ROOT, "data", "MIT_Indoor")


def discover_classes():
    """Return sorted class list from train split, or empty list."""
    train_dir = os.path.join(DATA_DIR, "train")
    if os.path.isdir(train_dir):
        return sorted(
            d for d in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, d))
        )
    # Fallback: read from test split
    test_dir = os.path.join(DATA_DIR, "test")
    if os.path.isdir(test_dir):
        return sorted(
            d for d in os.listdir(test_dir)
            if os.path.isdir(os.path.join(test_dir, d))
        )
    return []


def load_models():
    global baseline_model, hybrid_model, classes, hog_feature_dim

    classes = discover_classes()
    num_classes = max(len(classes), 1)   # avoid 0 for model init

    # ── CNN Baseline ──────────────────────────────────────────────────────────
    baseline_path = os.path.join(MODELS_DIR, "baseline.pth")
    if os.path.exists(baseline_path):
        try:
            baseline_model = CNNBaseline(num_classes)
            baseline_model.load_state_dict(
                torch.load(baseline_path, map_location=DEVICE)
            )
            baseline_model.to(DEVICE).eval()
            print(f"[INFO] Loaded CNN baseline from {baseline_path}")
        except Exception as e:
            print(f"[WARN] Could not load baseline model: {e}")
            baseline_model = None
    else:
        print(f"[WARN] baseline.pth not found at {baseline_path}")

    # ── Hybrid model ──────────────────────────────────────────────────────────
    hybrid_path = os.path.join(MODELS_DIR, "hybrid.pth")
    if os.path.exists(hybrid_path):
        try:
            # Probe feature dimension with a dummy image
            dummy = np.zeros((224, 224, 3), dtype=np.uint8)
            dummy_feat = stack_features(dummy)
            hog_feature_dim = dummy_feat.shape[0]

            hybrid_model = HybridCNN(hog_feature_dim, num_classes)
            hybrid_model.load_state_dict(
                torch.load(hybrid_path, map_location=DEVICE)
            )
            hybrid_model.to(DEVICE).eval()
            print(f"[INFO] Loaded Hybrid model (dim={hog_feature_dim}) from {hybrid_path}")
        except Exception as e:
            print(f"[WARN] Could not load hybrid model: {e}")
            hybrid_model = None
    else:
        print(f"[WARN] hybrid.pth not found at {hybrid_path}")


load_models()


# ── Helpers ───────────────────────────────────────────────────────────────────

def pil_to_cv2_rgb(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL image → uint8 RGB numpy array, resized to 224×224."""
    pil_img = pil_img.convert("RGB")
    pil_img = pil_img.resize((224, 224), Image.LANCZOS)
    return np.array(pil_img)


def fig_to_b64(fig: plt.Figure) -> str:
    """Render matplotlib figure to base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def ndarray_to_b64(arr: np.ndarray) -> str:
    """Convert uint8 HxWx3 numpy array to base64 PNG."""
    img = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def run_gradcam(model, input_tensor, target_layer, pred_idx, image_np_float):
    """Run Grad-CAM and return base64 overlay image."""
    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(pred_idx)]
    gs_cam = cam(input_tensor=input_tensor.clone().detach(), targets=targets)[0]
    cam_img = show_cam_on_image(image_np_float, gs_cam, use_rgb=True)
    return ndarray_to_b64(cam_img)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "device": str(DEVICE),
        "baseline_loaded": baseline_model is not None,
        "hybrid_loaded":   hybrid_model is not None,
        "num_classes":     len(classes),
        "classes":         classes,
    })


@app.route("/classes", methods=["GET"])
def get_classes():
    return jsonify({"classes": classes})


@app.route("/predict", methods=["POST"])
def predict():
    # ── Validate request ──────────────────────────────────────────────────────
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    model_type = request.form.get("model", "baseline")   # "baseline" | "hybrid"

    if model_type == "baseline" and baseline_model is None:
        return jsonify({"error": "Baseline model not loaded. Train it first (training/train_baseline.py)."}), 503

    if model_type == "hybrid" and hybrid_model is None:
        return jsonify({"error": "Hybrid model not loaded. Train it first (training/train_hybrid.py)."}), 503

    if not classes:
        return jsonify({"error": "No classes found. Make sure data/MIT_Indoor/train exists."}), 503

    try:
        # ── Read & preprocess image ───────────────────────────────────────────
        file = request.files["image"]
        pil_img = Image.open(file.stream)
        image_rgb = pil_to_cv2_rgb(pil_img)           # uint8 (224,224,3)
        image_float = image_rgb.astype(np.float32) / 255.0   # float [0,1]

        result = {"model": model_type, "classes": classes}

        # ── CNN Baseline ──────────────────────────────────────────────────────
        if model_type == "baseline":
            transform = get_transforms(train=False)
            input_tensor = transform(pil_img.convert("RGB").resize((224, 224)))
            input_tensor = input_tensor.unsqueeze(0).float().to(DEVICE)
            input_tensor.requires_grad_(True)

            with torch.no_grad():
                logits = baseline_model(input_tensor)

            probs    = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred_idx = int(np.argmax(probs))
            top5_idx = np.argsort(probs)[::-1][:5]

            result.update({
                "prediction":   classes[pred_idx],
                "confidence":   float(probs[pred_idx]),
                "top5": [
                    {"class": classes[i], "prob": float(probs[i])}
                    for i in top5_idx
                ],
            })

            # Grad-CAM
            try:
                cam_b64 = run_gradcam(
                    baseline_model,
                    input_tensor,
                    baseline_model.model.layer4[-1],
                    pred_idx,
                    image_float,
                )
                result["gradcam"] = cam_b64
            except Exception as e:
                result["gradcam_error"] = str(e)

        # ── Hybrid Model ──────────────────────────────────────────────────────
        elif model_type == "hybrid":
            feat = stack_features(image_rgb)               # shape (D,)
            input_tensor = feat.unsqueeze(0).float().to(DEVICE)

            with torch.no_grad():
                logits = hybrid_model(input_tensor)

            probs    = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred_idx = int(np.argmax(probs))
            top5_idx = np.argsort(probs)[::-1][:5]

            result.update({
                "prediction":   classes[pred_idx],
                "confidence":   float(probs[pred_idx]),
                "top5": [
                    {"class": classes[i], "prob": float(probs[i])}
                    for i in top5_idx
                ],
                "note": "Hybrid model uses HOG features — Grad-CAM not applicable.",
            })

        # Original image as b64
        result["original_image"] = ndarray_to_b64(image_rgb)

        return jsonify(result)

    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Explainable Hybrid CV System — API Server")
    print("  http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
