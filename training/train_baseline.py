# training/train_baseline.py
# ============================================================
# Baseline CNN Training — ResNet-18 Fine-Tuning (OPTIMIZED)
# ------------------------------------------------------------
# NEW OPTIMIZATIONS:
#   1. Batch size increased (64 → 128) for better GPU utilization
#   2. Mixed Precision Training (AMP) for faster computation
#   3. Epochs reduced (30 → 26) — avoids unnecessary late training
#   4. Epoch timing + average tracking
#
# Expected:
#   - Training time: ~15 min → ~7–9 min
#   - Accuracy: same or slightly improved
# ============================================================

import os
import sys
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data   import DataLoader, random_split
from torchvision        import models

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataset_loader import MITIndoorDataset, get_transforms


# ============================================================
# CONFIGURATION
# ============================================================
TRAIN_DIR   = "data/MIT_Indoor/train"
MODEL_OUT   = "models/baseline.pth"

NUM_EPOCHS  = 26
BATCH_SIZE  = 128
LR          = 1e-3
LR_MIN      = 1e-5
WEIGHT_DECAY= 1e-4
VAL_SPLIT   = 0.1
LABEL_SMOOTH= 0.1
NUM_CLASSES = 67

from torchvision.transforms.v2 import MixUp, CutMix
import random

# Instantiate outside the training loop (after num_classes is defined)
mixup_fn  = MixUp(alpha=0.4, num_classes=NUM_CLASSES)   # NUM_CLASSES must be 67
cutmix_fn = CutMix(alpha=1.0, num_classes=NUM_CLASSES)

def apply_mixup_cutmix(images, labels):
    """
    Applies MixUp (30%), CutMix (30%), or no mixing (40%) randomly.
    Returns images and soft labels.
    Only call during training — never during validation.
    """
    r = random.random()
    if r < 0.3:
        return mixup_fn(images, labels)   # labels become soft probability vectors
    elif r < 0.6:
        return cutmix_fn(images, labels)  # labels become soft probability vectors
    else:
        return images, labels              # labels remain integer class indices

# ============================================================
# MODEL
# ============================================================
def build_model(num_classes: int) -> nn.Module:
    from torchvision.models import ResNet18_Weights
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer3.parameters():
        param.requires_grad = True

    for param in model.layer4.parameters():
        param.requires_grad = True

    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )

    return model


# ============================================================
# DATA
# ============================================================
def get_dataloaders(num_workers: int = 8):  # 🔥 increased workers

    from torch.utils.data import Subset

    # Step 1: generate deterministic indices without instantiating transforms yet
    full_tmp  = MITIndoorDataset(root_dir=TRAIN_DIR, transform=None)
    n_total   = len(full_tmp)
    num_classes = len(full_tmp.classes)
    
    indices   = torch.randperm(n_total, generator=torch.Generator().manual_seed(42)).tolist()
    n_val     = int(n_total * VAL_SPLIT)
    train_idx = indices[n_val:]
    val_idx   = indices[:n_val]

    # Step 2: two completely separate dataset instances, each with the correct transform
    train_ds = MITIndoorDataset(root_dir=TRAIN_DIR, transform=get_transforms(train=True))
    val_ds   = MITIndoorDataset(root_dir=TRAIN_DIR, transform=get_transforms(train=False))

    # Step 3: subset each instance with its respective indices
    train_dataset = Subset(train_ds, train_idx)
    val_dataset   = Subset(val_ds, val_idx)
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, num_classes, full_tmp.classes


# ============================================================
# TRAINING
# ============================================================
def train(model, train_loader, val_loader, device):

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = optim.Adam(
        trainable_params,
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=LR_MIN
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

    # 🔥 Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda')

    best_val_acc   = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    total_start = time.time()
    epoch_times = []

    print(f"\n{'='*52}")
    print(f"  Training for {NUM_EPOCHS} epochs")
    print(f"  Train samples : {len(train_loader.dataset)}")
    print(f"  Val samples   : {len(val_loader.dataset)}")
    print(f"  LR schedule   : {LR} -> {LR_MIN} (cosine)")
    print(f"{'='*52}\n")

    for epoch in range(1, NUM_EPOCHS + 1):

        epoch_start = time.time()

        # ── TRAIN ─────────────────────────────────────────────
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            # 🔥 Mixed precision forward
            with torch.amp.autocast('cuda'):
                images, labels = apply_mixup_cutmix(images, labels)   # labels may become float soft vectors
                
                # NOTE: Train accuracy is suppressed ~5-10% when MixUp/CutMix are active (soft labels).
                # Val accuracy (clean labels) is the authoritative metric. This is expected behaviour.

                # When mixing occurred, labels should be float tensors of shape (B, 67)
                # When no mixing occurred, labels remain int tensors of shape (B,)
                # Both are valid inputs to CrossEntropyLoss

                outputs = model(images)
                loss = criterion(outputs, labels)   # CrossEntropyLoss handles both int and soft float targets

            # 🔥 Scaled backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = torch.argmax(outputs, dim=1)
            running_loss    += loss.item() * images.size(0)
            if labels.ndim > 1:
                labels_for_acc = torch.argmax(labels, dim=1)
            else:
                labels_for_acc = labels
            running_correct += (preds == labels_for_acc).sum().item()
            running_total   += images.size(0)

        train_loss = running_loss / running_total
        train_acc  = running_correct / running_total * 100

        # ── VALIDATION ────────────────────────────────────────
        model.eval()
        val_correct, val_total = 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)

                val_correct += (preds == labels).sum().item()
                val_total   += images.size(0)

        val_acc = val_correct / val_total * 100

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        improved = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            improved = "  <- best"

        # ── TIMING ────────────────────────────────────────────
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)

        print(
            f"Epoch [{epoch:2d}/{NUM_EPOCHS}] "
            f"Loss: {train_loss:.4f} | "
            f"Train: {train_acc:.1f}% | "
            f"Acc: {val_acc:.1f}% | "
            f"LR: {current_lr:.2e} | "
            f"Time: {epoch_time:.1f}s | "
            f"Avg: {avg_epoch_time:.1f}s"
            f"{improved}"
        )

    total_time = time.time() - total_start
    final_avg_time = sum(epoch_times) / len(epoch_times)

    print(f"\n{'='*52}")
    print(f"  Training complete")
    print(f"  Best Val Accuracy : {best_val_acc:.2f}%")
    print(f"  Total Time        : {total_time:.1f}s  ({total_time/60:.1f} min)")
    print(f"  Avg Epoch Time    : {final_avg_time:.2f}s")
    print(f"{'='*52}")

    return best_model_wts, best_val_acc

# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":

    print("=" * 60)
    print("  BASELINE CNN TRAINING — ResNet-18 Fine-Tuning")
    print("=" * 60)

    # ── Device ────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # ── Data ──────────────────────────────────────────────────
    train_loader, val_loader, num_classes, class_names = get_dataloaders()
    print(f"Number of classes : {num_classes}")

    # ── Model ─────────────────────────────────────────────────
    model = build_model(num_classes).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params  : {trainable:,} / {total:,}")

    # ── Train ─────────────────────────────────────────────────
    best_weights, best_val_acc = train(model, train_loader, val_loader, device)

    # ── Save ──────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    torch.save(best_weights, MODEL_OUT)

    print("\nBaseline CNN training completed and saved.")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")