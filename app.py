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
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from skimage.color import rgb2gray
from skimage.transform import resize as sk_resize
from skimage.feature import hog as sk_hog

from models.cnn_baseline import CNNBaseline
from models.hybrid_cnn import HybridCNN
from data.dataset_loader import get_transforms

app = Flask(__name__)
CORS(app)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

# HOG config — MUST match preprocessing/extract_hog_features.py
HOG_IMG_SIZE        = (128, 128)
HOG_ORIENTATIONS    = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)


def compute_hog(image_rgb: np.ndarray) -> np.ndarray:
    gray = rgb2gray(image_rgb)
    gray = sk_resize(gray, HOG_IMG_SIZE, anti_aliasing=True)
    feat = sk_hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys"
    )
    return feat.astype(np.float32)


baseline_model = None
hybrid_model   = None
classes        = []

MODELS_DIR = os.path.join(ROOT, "models")
DATA_DIR   = os.path.join(ROOT, "data", "MIT_Indoor")


def discover_classes():
    for split in ("train", "test"):
        d = os.path.join(DATA_DIR, split)
        if os.path.isdir(d):
            return sorted(x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x)))
    return []


def load_models():
    global baseline_model, hybrid_model, classes
    classes     = discover_classes()
    num_classes = max(len(classes), 1)

    baseline_path = os.path.join(MODELS_DIR, "baseline.pth")
    if os.path.exists(baseline_path):
        try:
            baseline_model = CNNBaseline(num_classes)
            baseline_model.load_state_dict(torch.load(baseline_path, map_location=DEVICE))
            baseline_model.to(DEVICE).eval()
            print(f"[INFO] Loaded CNN baseline ({num_classes} classes)")
        except Exception as e:
            print(f"[WARN] Could not load baseline model: {e}")

    hybrid_path = os.path.join(MODELS_DIR, "hybrid.pth")
    if os.path.exists(hybrid_path):
        try:
            dummy_feat = compute_hog(np.zeros((224, 224, 3), dtype=np.uint8))
            feat_dim   = dummy_feat.shape[0]
            hybrid_model = HybridCNN(feat_dim, num_classes)
            hybrid_model.load_state_dict(torch.load(hybrid_path, map_location=DEVICE))
            hybrid_model.to(DEVICE).eval()
            print(f"[INFO] Loaded Hybrid model (HOG dim={feat_dim}, {num_classes} classes)")
        except Exception as e:
            print(f"[WARN] Could not load hybrid model: {e}")


load_models()


def pil_to_rgb224(pil_img):
    return np.array(pil_img.convert("RGB").resize((224, 224), Image.LANCZOS))


def ndarray_to_b64(arr):
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def run_gradcam(model, input_tensor, target_layer, pred_idx, image_float):
    cam     = GradCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(pred_idx)]
    gs_cam  = cam(input_tensor=input_tensor.clone().detach(), targets=targets)[0]
    return ndarray_to_b64(show_cam_on_image(image_float, gs_cam, use_rgb=True))


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok", "device": str(DEVICE),
        "baseline_loaded": baseline_model is not None,
        "hybrid_loaded":   hybrid_model   is not None,
        "num_classes": len(classes), "classes": classes,
    })


@app.route("/classes", methods=["GET"])
def get_classes():
    return jsonify({"classes": classes})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    model_type = request.form.get("model", "baseline")

    if model_type == "baseline" and baseline_model is None:
        return jsonify({"error": "Baseline model not loaded. Run: python training/train_baseline.py"}), 503
    if model_type == "hybrid" and hybrid_model is None:
        return jsonify({"error": "Hybrid model not loaded. Run: python training/train_hybrid.py"}), 503
    if not classes:
        return jsonify({"error": "No classes found. Check data/MIT_Indoor/train exists."}), 503

    try:
        pil_img   = Image.open(request.files["image"].stream)
        image_rgb = pil_to_rgb224(pil_img)
        image_f   = image_rgb.astype(np.float32) / 255.0
        result    = {"model": model_type, "classes": classes}

        if model_type == "baseline":
            transform    = get_transforms(train=False)
            input_tensor = transform(pil_img.convert("RGB").resize((224, 224)))
            input_tensor = input_tensor.unsqueeze(0).float().to(DEVICE)
            input_tensor.requires_grad_(True)

            with torch.no_grad():
                logits = baseline_model(input_tensor)

            probs    = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred_idx = int(np.argmax(probs))
            top5_idx = np.argsort(probs)[::-1][:5]
            result.update({
                "prediction": classes[pred_idx],
                "confidence": float(probs[pred_idx]),
                "top5": [{"class": classes[i], "prob": float(probs[i])} for i in top5_idx],
            })
            try:
                result["gradcam"] = run_gradcam(
                    baseline_model, input_tensor,
                    baseline_model.model.layer4[-1], pred_idx, image_f)
            except Exception as e:
                result["gradcam_error"] = str(e)

        elif model_type == "hybrid":
            hog_feat     = compute_hog(image_rgb)
            input_tensor = torch.tensor(hog_feat).unsqueeze(0).float().to(DEVICE)

            with torch.no_grad():
                logits = hybrid_model(input_tensor)

            probs    = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred_idx = int(np.argmax(probs))
            top5_idx = np.argsort(probs)[::-1][:5]
            result.update({
                "prediction": classes[pred_idx],
                "confidence": float(probs[pred_idx]),
                "top5": [{"class": classes[i], "prob": float(probs[i])} for i in top5_idx],
                "note": "Hybrid model uses flat HOG features — Grad-CAM not applicable.",
            })

        result["original_image"] = ndarray_to_b64(image_rgb)
        return jsonify(result)

    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Explainable Hybrid CV System — API Server")
    print("  http://localhost:5000")
    print("="*60 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
