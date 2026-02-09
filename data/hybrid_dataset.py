# data/hybrid_dataset.py
# ============================================================
# Hybrid Dataset (Classical Features Only)
# ------------------------------------------------------------
# - Loads precomputed HOG features from .npz files
# - NO image loading
# - NO disk access during training loops
# - Used by hybrid (feature-based) classifier
# ============================================================

import numpy as np
import torch
from torch.utils.data import Dataset


class HybridDataset(Dataset):
    """
    Dataset for classical feature-based models (HOG).

    Expected files:
    - data/hog_features_train.npz
    - data/hog_features_test.npz

    Each .npz must contain:
    - features : (N, D) float32
    - labels   : (N,) int
    """

    def __init__(self, split: str = "train"):
        assert split in {"train", "test"}, "split must be 'train' or 'test'"

        npz_path = f"data/hog_features_{split}.npz"

        # ----------------------------------------------------
        # Load precomputed features ONCE
        # ----------------------------------------------------
        data = np.load(npz_path)

        self.features = torch.from_numpy(
            data["features"]
        ).float()

        self.labels = torch.from_numpy(
            data["labels"]
        ).long()

        # Store unique classes for evaluation scripts
        self.classes = torch.unique(self.labels).tolist()

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        """
        Returns:
        - feature vector (D,)
        - label (int)
        """
        return self.features[idx], self.labels[idx]