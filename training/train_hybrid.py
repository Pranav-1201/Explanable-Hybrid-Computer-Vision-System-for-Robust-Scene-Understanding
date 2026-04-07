# training/train_hybrid.py
# ============================================================
# IMPROVED: Hybrid Model Training
# ------------------------------------------------------------
# Key improvements over original:
#   1. Cosine annealing LR schedule (not fixed LR)
#   2. Validation split for honest early stopping
#      (original stopped on train loss → overfit)
#   3. Best model checkpoint by val accuracy
#   4. Label smoothing in CrossEntropyLoss
#   5. Higher patience — was 3, now 10
#   6. Warmup epochs before LR decay
# ============================================================

import os
import sys
import warnings
import time
import numpy as np

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data.hybrid_dataset import HybridDataset
from models.hybrid_cnn import HybridCNN


# ============================================================
# EARLY STOPPING (on VALIDATION accuracy, not train loss)
# ============================================================
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_score = None
        self.counter    = 0

    def step(self, val_acc: float) -> bool:
        if self.best_score is None or val_acc > self.best_score + self.min_delta:
            self.best_score = val_acc
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


# ============================================================
# TRAIN
# ============================================================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ----------------------------------------------------------
    # DATASET — split train into 90% train / 10% val
    # ----------------------------------------------------------
    full_dataset = HybridDataset(split="train")
    n_total      = len(full_dataset)
    n_val        = max(1, int(0.10 * n_total))
    n_train      = n_total - n_val

    train_dataset, val_dataset = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    feature_dim  = full_dataset.features.shape[1]
    num_classes  = len(full_dataset.classes)

    print(f"Feature dim   : {feature_dim}")
    print(f"Num classes   : {num_classes}")
    print(f"Train samples : {n_train}")
    print(f"Val samples   : {n_val}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0
    )

    # ----------------------------------------------------------
    # MODEL
    # ----------------------------------------------------------
    model = HybridCNN(feature_dim, num_classes).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params  : {total_params:,}")

    # Label smoothing helps prevent overconfident overfitting
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-3    # stronger L2 than original (was 1e-4)
    )

    # Cosine annealing: LR warms up for 5 epochs then decays
    epochs = 100
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-4,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.05,       # 5% warmup
        anneal_strategy="cos"
    )

    # ----------------------------------------------------------
    # TRAINING LOOP
    # ----------------------------------------------------------
    early_stopper = EarlyStopping(patience=10)
    best_val_acc  = 0.0
    best_epoch    = 0

    os.makedirs("models", exist_ok=True)
    training_start = time.time()

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        running_loss = 0.0
        correct = total = 0

        for features, labels in train_loader:
            features = features.to(device)
            labels   = labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss    = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            preds    = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc  = 100.0 * correct / total

        # --- Validate ---
        model.eval()
        val_correct = val_total = 0

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels   = labels.to(device)
                outputs  = model(features)
                preds    = torch.argmax(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)

        val_acc = 100.0 * val_correct / val_total
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch [{epoch+1:3d}/{epochs}] "
            f"Loss: {train_loss:.4f} | "
            f"Train: {train_acc:.1f}% | "
            f"Val: {val_acc:.1f}% | "
            f"LR: {current_lr:.2e}"
        )

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch + 1
            torch.save(model.state_dict(), "models/hybrid.pth")

        # Early stopping on VAL accuracy
        if early_stopper.step(val_acc):
            print(f"\n[EARLY STOP] No improvement for {early_stopper.patience} epochs.")
            break

    total_time = time.time() - training_start

    print(f"\n{'='*50}")
    print(f"Best Val Accuracy : {best_val_acc:.2f}% (epoch {best_epoch})")
    print(f"Total Train Time  : {total_time:.1f}s")
    print(f"Saved: models/hybrid.pth")
    print(f"{'='*50}")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    train()