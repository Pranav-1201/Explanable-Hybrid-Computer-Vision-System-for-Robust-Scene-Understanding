# training/train_hybrid.py
# ============================================================
# Hybrid Model Training Script (HOG Features)
# ------------------------------------------------------------
# - Trains an MLP on precomputed HOG features
# - Extremely fast (seconds, not minutes)
# - No image loading, no CPU bottlenecks
# ============================================================

import os
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import time

from data.hybrid_dataset import HybridDataset
from models.hybrid_cnn import HybridCNN

# ------------------------------------------------------------
# EARLY STOPPING
# ------------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0

    def step(self, loss):
        if self.best_loss is None or loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

# ------------------------------------------------------------
# TRAIN FUNCTION
# ------------------------------------------------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------
    train_dataset = HybridDataset(split="train")

    feature_dim = train_dataset.features.shape[1]
    num_classes = len(train_dataset.classes)

    print(f"Feature dimension: {feature_dim}")
    print(f"Number of classes: {num_classes}")
    print(f"Training samples : {len(train_dataset)}")

    # --------------------------------------------------------
    # DATALOADER
    # --------------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=256,          # Large batch = GPU efficient
        shuffle=True,
        num_workers=0            # No disk I/O, no workers needed
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------
    model = HybridCNN(feature_dim, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )

    # --------------------------------------------------------
    # TRAIN LOOP
    # --------------------------------------------------------
    epochs = 50
    model.train()
    early_stopper = EarlyStopping(patience=3)

    # --------------------------------------------------------
    # Training Time Tracking
    # --------------------------------------------------------
    epoch_times = []
    training_start_time = time.time()

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        # GPU-safe timing (hybrid also runs on CUDA)
        if device.type == "cuda":
            torch.cuda.synchronize()

        epoch_start_time = time.time()

        print(f"\nEpoch [{epoch+1}/{epochs}]")

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total

        print(f"Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")
        if early_stopper.step(epoch_loss):
            print("🛑 Early stopping triggered")
            break
        # --------------------------------------------------------
        # Epoch Timing
        # --------------------------------------------------------
        if device.type == "cuda":
            torch.cuda.synchronize()

        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)

        avg_epoch_time = sum(epoch_times) / len(epoch_times)

        print(
            f"[TIME] Epoch {epoch+1}/{epochs} | "
            f"Epoch Time: {epoch_time:.2f}s | "
            f"Avg Epoch Time: {sum(epoch_times)/max(1, len(epoch_times)):.2f}s"
        )

    # --------------------------------------------------------
    # Total Training Time Summary
    # --------------------------------------------------------
    total_time = time.time() - training_start_time

    print("\n================ TRAINING TIME SUMMARY ================")
    print(f"Total Training Time : {total_time:.2f} seconds")
    print(f"Avg Epoch Time      : {sum(epoch_times)/len(epoch_times):.2f} seconds")
    print("=======================================================")

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/hybrid.pth")
    print("\nHybrid model training completed and saved.")


# ------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------
if __name__ == "__main__":
    train()