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

    from utils.checkpoint import load_checkpoint
    model = load_checkpoint("models/baseline.pth", model, device=device)

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
# PHASE 2 EVALUATION
# ------------------------------------------------------------
def evaluate_phase2():
    print("\n================ PHASE 2 (ResNet-50) =================")
    checkpoint_path = "models/phase2_best.pth"
    if not os.path.exists(checkpoint_path):
        print("Model not found. Skipping Phase 2 evaluation.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from models.cnn_baseline import CNNBaseline
    from inference.tta import tta_predict, single_predict
    from data.dataset_loader import MITIndoorDataset
    from utils.checkpoint import load_checkpoint

    test_dataset = MITIndoorDataset(root_dir="data/MIT_Indoor/test", transform=None)
    num_classes = len(test_dataset.classes)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    backbone = checkpoint.get('backbone', 'resnet50_places365')
    
    raw_model = CNNBaseline(num_classes, backbone=backbone).to(device)
    raw_model = load_checkpoint(checkpoint_path, raw_model, device=device)
    raw_model.eval()

    use_ema = checkpoint.get('ema_state') is not None
    if use_ema:
        ema_model = CNNBaseline(num_classes, backbone=backbone).to(device)
        ema_model = load_checkpoint(checkpoint_path, ema_model, load_ema=True, device=device)
        ema_model.eval()
    else:
        ema_model = None

    results = {
        'raw': {'correct': 0, 'total': 0},
        'raw_tta': {'correct': 0, 'total': 0},
        'ema': {'correct': 0, 'total': 0},
        'ema_tta': {'correct': 0, 'total': 0},
    }

    print("Evaluating Phase 2 models... (this may take a while with TTA)")
    for i in range(len(test_dataset)):
        pil_img, label = test_dataset[i]
        
        # Raw single
        probs_raw = single_predict(raw_model, pil_img, device)
        if int(torch.argmax(probs_raw)) == label: results['raw']['correct'] += 1
        results['raw']['total'] += 1

        # Raw TTA
        probs_raw_tta = tta_predict(raw_model, pil_img, device)
        if int(torch.argmax(probs_raw_tta)) == label: results['raw_tta']['correct'] += 1
        results['raw_tta']['total'] += 1

        if use_ema:
            # EMA single
            probs_ema = single_predict(ema_model, pil_img, device)
            if int(torch.argmax(probs_ema)) == label: results['ema']['correct'] += 1
            results['ema']['total'] += 1

            # EMA TTA
            probs_ema_tta = tta_predict(ema_model, pil_img, device)
            if int(torch.argmax(probs_ema_tta)) == label: results['ema_tta']['correct'] += 1
            results['ema_tta']['total'] += 1

    print("--- Phase 2 Summary ---")
    print(f"Raw Model Accuracy      : {results['raw']['correct'] / results['raw']['total'] * 100:.2f}%")
    print(f"Raw Model + TTA Accuracy: {results['raw_tta']['correct'] / results['raw_tta']['total'] * 100:.2f}%")
    
    if use_ema:
        print(f"EMA Model Accuracy      : {results['ema']['correct'] / results['ema']['total'] * 100:.2f}%")
        print(f"EMA Model + TTA Accuracy: {results['ema_tta']['correct'] / results['ema_tta']['total'] * 100:.2f}%")

# ------------------------------------------------------------
# FUSION MODEL EVALUATION
# ------------------------------------------------------------
def evaluate_fusion():
    print("\n================ HYBRID FUSION MODEL =================")
    checkpoint_path = "models/fusion_best.pth"
    if not os.path.exists(checkpoint_path):
        print("Fusion model not found. Skipping.")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Needs training/train_fusion.py to be importable for FusionDataset
    from training.train_fusion import FusionDataset
    from models.hybrid_fusion import HybridFusion
    from sklearn.metrics import f1_score
    
    test_ds = FusionDataset('test')
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    
    num_classes = 67
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = HybridFusion(cnn_dim=2048, hog_dim=512, num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint['fusion_state'])
    model.eval()

    y_true, y_pred, y_pred_ablated = [], [], []

    with torch.no_grad():
        for cnn_feat, hog_feat, labels in test_loader:
            cnn_feat, hog_feat = cnn_feat.to(device), hog_feat.to(device)
            
            # Normal forward
            logits = model(cnn_feat, hog_feat)
            preds = torch.argmax(logits, dim=1)
            
            # Ablated forward
            logits_ablated = model.ablate_hog(cnn_feat)
            preds_ablated = torch.argmax(logits_ablated, dim=1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_pred_ablated.extend(preds_ablated.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    acc_ablated = accuracy_score(y_true, y_pred_ablated)
    
    print(f"Fusion (CNN + HOG) accuracy: {acc * 100:.2f}%")
    print(f"Fusion (CNN only, HOG=0):    {acc_ablated * 100:.2f}%")
    print(f"HOG contribution:            +{(acc - acc_ablated) * 100:.2f}%")
    
    # Hardest classes
    f1_scores = f1_score(y_true, y_pred, average=None)
    hardest_idx = f1_scores.argsort()[:5]
    print("\nTop-5 Hardest Classes (Lowest F1):")
    # For class names we need MITIndoorDataset
    from data.dataset_loader import MITIndoorDataset
    dataset = MITIndoorDataset("data/MIT_Indoor/test", transform=None)
    classes = dataset.classes
    for idx in hardest_idx:
        print(f"  {classes[idx]}: F1 = {f1_scores[idx]:.3f}")




# ------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------
if __name__ == "__main__":
    evaluate_baseline()
    evaluate_hybrid()
    evaluate_transfer_model()
    evaluate_phase2()
    evaluate_fusion()