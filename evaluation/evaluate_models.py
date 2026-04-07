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
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from data.dataset_loader import MITIndoorDataset, get_transforms, get_test_loader
from data.hybrid_dataset import HybridDataset

from models.cnn_baseline import CNNBaseline
from models.hybrid_cnn import HybridCNN

from torchvision import models



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
    valid_labels = list(range(num_classes))

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
        pin_memory=(device.type == "cuda")
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------
    model = CNNBaseline(num_classes).to(device)

    checkpoint = torch.load("models/baseline.pth", map_location=device)

    # Fix key mismatch (add "model." prefix)
    new_state_dict = {}
    for k, v in checkpoint.items():
        if not k.startswith("model."):
            new_state_dict["model." + k] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)

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
        labels=valid_labels,
        target_names=class_names,
        zero_division=0
    ))



# ------------------------------------------------------------
# TRANSFER MODEL EVALUATION
# ------------------------------------------------------------
def evaluate_transfer_model():

    print("\n================ RESNET50 TRANSFER =================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Faster evaluation batch for RTX 4060
    test_loader = get_test_loader(batch_size=64)

    num_classes = len(test_loader.dataset.classes)

    # Recreate architecture exactly like training
    model = models.resnet50(weights=None)

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    )

    checkpoint = torch.load("models/transfer_resnet50.pth", map_location=device)
    model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)

            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)

    print(f"Accuracy: {acc*100:.2f}%")



# ------------------------------------------------------------
# HYBRID MODEL EVALUATION
# ------------------------------------------------------------
def evaluate_hybrid():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------
    test_dataset = HybridDataset(split="test")

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0
    )

    feature_dim = test_dataset.features.shape[1]
    num_classes = len(test_dataset.classes)

    class_names = [str(c) for c in test_dataset.classes]
    valid_labels = list(range(num_classes))

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------
    model = HybridCNN(feature_dim, num_classes).to(device)

    checkpoint = torch.load("models/hybrid.pth", map_location=device)
    model.load_state_dict(checkpoint)

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

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=valid_labels)

    print("\n================ HYBRID MODEL =================")
    print(f"Accuracy: {acc * 100:.2f}%")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=valid_labels,
        target_names=class_names,
        zero_division=0
    ))



# ------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------
if __name__ == "__main__":
    evaluate_baseline()
    evaluate_hybrid()
    evaluate_transfer_model()