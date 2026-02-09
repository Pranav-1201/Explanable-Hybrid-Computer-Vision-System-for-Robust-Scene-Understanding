print(">>> RUNNING train_cnn_features.py <<<")

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------- DATASET ----------------
class FeatureDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.X = torch.tensor(data["features"], dtype=torch.float32)
        self.y = torch.tensor(data["labels"], dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ---------------- TRAIN ----------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = FeatureDataset("data/cnn_features_train.npz")
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    num_classes = len(torch.unique(dataset.y))
    model = nn.Linear(512, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 10

    # ---------------- TIMERS ----------------
    total_start_time = time.time()
    epoch_times = []
    batch_times = []

    # ---------------- TRAIN LOOP ----------------
    for epoch in range(epochs):
        epoch_start_time = time.time()
        running_loss = 0.0

        print(f"\nEpoch {epoch+1}/{epochs}")

        for i, (X, y) in enumerate(loader):
            batch_start_time = time.time()

            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)

            print(
                f"  Batch {i+1}/{len(loader)} | "
                f"Loss: {loss.item():.4f} | "
                f"Batch Time: {batch_time:.3f}s"
            )

        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)

        print(
            f"Epoch {epoch+1} finished | "
            f"Avg Loss: {running_loss / len(loader):.4f} | "
            f"Epoch Time: {epoch_time:.2f}s"
        )

    # ---------------- FINAL STATS ----------------
    total_time = time.time() - total_start_time

    print("\n================ TRAINING SUMMARY ================")
    print(f"Total training time       : {total_time:.2f} seconds")
    print(f"Average time per epoch    : {np.mean(epoch_times):.2f} seconds")
    print(f"Average time per batch    : {np.mean(batch_times):.3f} seconds")
    print("==================================================")

    torch.save(model.state_dict(), "models/cnn_feature_classifier.pth")
    print("Model saved: models/cnn_feature_classifier.pth")

if __name__ == "__main__":
    train()
