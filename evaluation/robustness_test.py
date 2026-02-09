# evaluation/robustness_test.py
# ============================================================
# Robustness Testing (Image-Based Models Only)
# ------------------------------------------------------------
# - Evaluates Baseline CNN under image corruptions
# - Hybrid model is intentionally excluded
#   (classical features are precomputed & non-image-based)
# ============================================================

import os
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader

from PIL import Image
from data.dataset_loader import MITIndoorDataset, get_transforms
from models.cnn_baseline import CNNBaseline


# ------------------------------------------------------------
# IMAGE CORRUPTION FUNCTIONS
# ------------------------------------------------------------
def add_noise(image):
    noise = np.random.normal(0, 25, image.shape).astype(np.float32)
    return np.clip(image + noise, 0, 255).astype(np.uint8)


def add_blur(image):
    return cv2.GaussianBlur(image, (7, 7), 0)


def add_occlusion(image):
    h, w, _ = image.shape
    image = image.copy()
    image[h // 4:h // 2, w // 4:w // 2] = 0
    return image


# ------------------------------------------------------------
# ROBUSTNESS DATASET WRAPPER
# ------------------------------------------------------------
class CorruptedDataset(MITIndoorDataset):
    """
    Wraps MITIndoorDataset to apply corruption
    BEFORE torchvision transforms.
    """

    def __init__(self, root_dir, corruption_fn):
        super().__init__(root_dir, transform=None)
        self.corruption_fn = corruption_fn
        self.post_transform = get_transforms(train=False)

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)

        # PIL -> numpy
        image = np.array(image)

        # Apply corruption
        image = self.corruption_fn(image)

        # Back to PIL then transform (FIX)
        image = self.post_transform(Image.fromarray(image))

        return image, label


# ------------------------------------------------------------
# EVALUATION FUNCTION
# ------------------------------------------------------------
def evaluate(model, loader, device):
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


# ------------------------------------------------------------
# MAIN TEST
# ------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------
    # LOAD TRAIN DATASET (FOR CLASS CONSISTENCY)
    # --------------------------------------------------------
    train_dataset = MITIndoorDataset(
        root_dir="data/MIT_Indoor/train",
        transform=None
    )
    num_classes = len(train_dataset.classes)

    # --------------------------------------------------------
    # LOAD BASELINE MODEL
    # --------------------------------------------------------
    model = CNNBaseline(num_classes)
    model.load_state_dict(
        torch.load("models/baseline.pth", map_location=device)
    )
    model.to(device)

    test_root = "data/MIT_Indoor/test"

    corruptions = {
        "Clean": lambda x: x,
        "Gaussian Noise": add_noise,
        "Gaussian Blur": add_blur,
        "Occlusion": add_occlusion,
    }

    print("\n========== BASELINE CNN ROBUSTNESS ==========")

    for name, fn in corruptions.items():
        dataset = CorruptedDataset(test_root, fn)

        loader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        acc = evaluate(model, loader, device)
        print(f"{name:15s}: Accuracy = {acc * 100:.2f}%")

    print("\nNOTE:")
    print("Hybrid HOG model robustness is not evaluated,")
    print("because classical features are precomputed and")
    print("not directly affected by image-time corruptions.")


# ------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------
if __name__ == "__main__":
    main()