import os
import sys
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader
from torchvision import models

from utils.transforms import get_transforms
from data.dataset_loader import MITIndoorDataset

# ---------------- SETTINGS ----------------
DATA_DIR = "data/MIT_Indoor/train"
SAVE_PATH = "data/cnn_features_train.npz"
BATCH_SIZE = 32

# ---------------- MAIN ----------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset
    dataset = MITIndoorDataset(
        root_dir=DATA_DIR,
        transform=get_transforms(train=False)  # no augmentation
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # Load pretrained ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()  # REMOVE classifier
    model.to(device)
    model.eval()

    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Extracting CNN features"):
            images = images.to(device)
            features = model(images)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    os.makedirs("data", exist_ok=True)
    np.savez(SAVE_PATH, features=all_features, labels=all_labels)

    print(f"Saved CNN features to {SAVE_PATH}")
    print("Feature shape:", all_features.shape)

if __name__ == "__main__":
    main()
