# evaluation/evaluate_models.py
# ============================================================
# Model Evaluation Script
# ------------------------------------------------------------
# - Evaluates Baseline CNN (image-based)
# - Evaluates Hybrid MLP (HOG-based)
# - Produces accuracy, confusion matrix, classification report
# ============================================================

import os
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from data.dataset_loader import MITIndoorDataset, get_transforms
from data.hybrid_dataset import HybridDataset
from models.cnn_baseline import CNNBaseline
from models.hybrid_cnn import HybridCNN


# ------------------------------------------------------------
# BASELINE CNN EVALUATION
# ------------------------------------------------------------
def evaluate_baseline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------
    # LOAD TRAIN DATASET (FOR CLASS CONSISTENCY)
    # --------------------------------------------------------
    train_dataset = MITIndoorDataset(
        root_dir="data/MIT_Indoor/train",
        transform=None
    )

    num_classes = len(train_dataset.classes)
    class_names = train_dataset.classes
    valid_labels = list(range(num_classes))  # 🔑 KEY FIX

    # --------------------------------------------------------
    # TEST DATASET & LOADER
    # --------------------------------------------------------
    test_dataset = MITIndoorDataset(
        root_dir="data/MIT_Indoor/test",
        transform=get_transforms(train=False)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------
    model = CNNBaseline(num_classes)
    model.load_state_dict(torch.load("models/baseline.pth", map_location=device))
    model.to(device)
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=valid_labels)

    print("\n================ BASELINE CNN =================")
    print(f"Accuracy: {acc * 100:.2f}%")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=valid_labels,          # 🔑 KEY FIX
        target_names=class_names,
        zero_division=0
    ))


# ------------------------------------------------------------
# HYBRID MODEL EVALUATION
# ------------------------------------------------------------
def evaluate_hybrid():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = HybridDataset(split="test")

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0
    )

    feature_dim = test_dataset.features.shape[1]
    num_classes = len(test_dataset.classes)

    model = HybridCNN(feature_dim, num_classes)
    model.load_state_dict(torch.load("models/hybrid.pth", map_location=device))
    model.to(device)
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print("\n================ HYBRID MODEL =================")
    print(f"Accuracy: {acc * 100:.2f}%")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        zero_division=0
    ))


# ------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------
if __name__ == "__main__":
    evaluate_baseline()
    evaluate_hybrid()