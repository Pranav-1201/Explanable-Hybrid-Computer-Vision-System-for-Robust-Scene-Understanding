import sys
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from models.hybrid_cnn import HybridCNN
from classical_features.feature_stack import stack_features

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load classes
classes = sorted(os.listdir("data/MIT_Indoor/test"))

# Load model
model = HybridCNN(len(classes))
model.load_state_dict(torch.load("models/hybrid.pth", weights_only=True))
model.to(device)
model.eval()

# Load image (CHANGE THIS PATH FOR DEMO)
# Automatically pick one test image
test_root = "data/MIT_Indoor/test"
cls = os.listdir(test_root)[0]
img_dir = os.path.join(test_root, cls)
img_name = os.listdir(img_dir)[0]
img_path = os.path.join(img_dir, img_name)


image_bgr = cv2.imread(img_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
image_rgb = cv2.resize(image_rgb, (224, 224))

# Prepare input
stacked = stack_features(image_rgb)
input_tensor = stacked.unsqueeze(0).float().to(device)

# Prediction
with torch.no_grad():
    outputs = model(input_tensor)
    pred_idx = torch.argmax(outputs, dim=1).item()

print("Predicted Scene:", classes[pred_idx])

# Grad-CAM
target_layer = model.model.layer4[-1]
cam = GradCAM(model=model, target_layers=[target_layer])
targets = [ClassifierOutputTarget(pred_idx)]

grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
cam_image = show_cam_on_image(image_rgb / 255.0, grayscale_cam, use_rgb=True)

# Display
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(image_rgb)
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Grad-CAM Explanation")
plt.imshow(cam_image)
plt.axis("off")

plt.show()
