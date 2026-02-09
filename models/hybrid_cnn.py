# models/hybrid_cnn.py
# ============================================================
# Hybrid Model (Classical Features Only)
# ------------------------------------------------------------
# - Consumes precomputed HOG feature vectors
# - NO CNN backbone
# - Implemented as a clean MLP classifier
# - Extremely fast and stable
# ============================================================

import torch
import torch.nn as nn


class HybridCNN(nn.Module):
    """
    Hybrid classifier operating on classical features (HOG).

    Input:
    - Feature vector of shape (D,)

    Output:
    - Class logits of shape (num_classes,)
    """

    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()

        # ----------------------------------------------------
        # Fully Connected Classification Head
        # ----------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: feature tensor (B, D)

        Returns:
            logits (B, num_classes)
        """
        return self.classifier(x)