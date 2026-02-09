# explainability/gradcam_explain.py
# ============================================================
# Grad-CAM Explainability (Baseline CNN Only)
# ------------------------------------------------------------
# - Generates Grad-CAM visualizations for ResNet-18 baseline
# - Hybrid model is intentionally excluded
# - Output images are saved for reports / PPTs
# ============================================================

import os
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from data.dataset_loader import MITIndoorDataset, get_transforms
from models.cnn_baseline import CNNBaseline


# ------------------------------------------------------------
# SETUP
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_root = "data/MIT_Indoor/test"

# ------------------------------------------------------------
# FIX: derive class count from TRAIN dataset (not test)
# ------------------------------------------------------------
train_dataset = MITIndoorDataset(
    root_dir="data/MIT_Indoor/train",
    transform=None
)
num_classes = len(train_dataset.classes)

os.makedirs("results", exist_ok=True)


# ------------------------------------------------------------
# LOAD BASELINE MODEL
# ------------------------------------------------------------
model = CNNBaseline(num_classes)
model.load_state_dict(
    torch.load("models/baseline.pth", map_location=device)
)
model.to(device)
model.eval()


# ------------------------------------------------------------
# LOAD ONE TEST IMAGE
# ------------------------------------------------------------
dataset = MITIndoorDataset(
    root_dir=test_root,
    transform=get_transforms(train=False)
)

image_tensor, label = dataset[0]
input_tensor = image_tensor.unsqueeze(0).to(device)
input_tensor.requires_grad_(True)

# Convert tensor → numpy RGB for visualization
image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())


# ------------------------------------------------------------
# GRAD-CAM
# ------------------------------------------------------------
target_layer = model.model.layer4[-1]

cam = GradCAM(
    model=model,
    target_layers=[target_layer]
)

targets = [ClassifierOutputTarget(int(label))]
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

cam_image = show_cam_on_image(image_np, grayscale_cam, use_rgb=True)


# ------------------------------------------------------------
# DISPLAY & SAVE
# ------------------------------------------------------------
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image_np)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Grad-CAM (Baseline CNN)")
plt.imshow(cam_image)
plt.axis("off")

plt.tight_layout()
plt.show()

cv2.imwrite("results/original.jpg", (image_np * 255).astype(np.uint8)[..., ::-1])
cv2.imwrite("results/gradcam_baseline.jpg", cam_image[..., ::-1])

print("Grad-CAM images saved to /results")