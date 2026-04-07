# data/hybrid_dataset.py
# ============================================================
# IMPROVED: Hybrid Dataset
# ------------------------------------------------------------
# Key fix: Stats file name tied to feature file to avoid
# stale stats when features are regenerated.
# Also prints feature dim on load for easier debugging.
# ============================================================

import numpy as np
import torch
import os
from torch.utils.data import Dataset


class HybridDataset(Dataset):
    """
    Loads precomputed feature vectors (.npz) for the hybrid model.

    Expected files:
      data/hog_features_train.npz
      data/hog_features_test.npz

    Each must contain:
      features : (N, D) float32
      labels   : (N,)  int64
    """

    def __init__(self, split: str = "train"):
        assert split in {"train", "test"}

        npz_path = f"data/hog_features_{split}.npz"

        if not os.path.exists(npz_path):
            raise FileNotFoundError(
                f"Feature file not found: {npz_path}\n"
                "Run: python preprocessing/extract_hog_features.py"
            )

        data     = np.load(npz_path)
        features = data["features"].astype(np.float32)

        print(f"[HybridDataset] Loaded {split}: {features.shape}")

        # ----------------------------------------------------------
        # Normalization — always computed from TRAIN set
        # Stats file is tied to train features so it invalidates
        # when train features change.
        # ----------------------------------------------------------
        stats_path = "data/hog_feature_stats.npz"

        if not os.path.exists(stats_path):
            print("[HybridDataset] Computing normalization stats from train set...")
            train_data     = np.load("data/hog_features_train.npz")
            train_features = train_data["features"].astype(np.float32)
            mean = train_features.mean(axis=0)
            std  = train_features.std(axis=0) + 1e-6
            np.savez(stats_path, mean=mean, std=std)
            print(f"[HybridDataset] Stats saved: {stats_path}")

        stats    = np.load(stats_path)
        mean     = stats["mean"]
        std      = stats["std"]
        features = (features - mean) / std

        self.features = torch.from_numpy(features).float()
        self.labels   = torch.from_numpy(data["labels"]).long()

        # Class list: 67 fixed indices for MIT Indoor
        self.classes = list(range(67))

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]