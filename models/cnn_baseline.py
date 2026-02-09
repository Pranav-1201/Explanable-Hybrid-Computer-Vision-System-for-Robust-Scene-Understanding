# models/cnn_baseline.py
# ============================================================
# Baseline CNN Model (ResNet-18)
# ------------------------------------------------------------
# - ImageNet-pretrained backbone
# - Fine-tunes ONLY the classification head
# - Stable, fast, and Grad-CAM compatible
# ============================================================

import torch
import torch.nn as nn
from torchvision import models


class CNNBaseline(nn.Module):
    """
    Baseline CNN for MIT Indoor Scene Classification.

    Architecture:
    - ResNet-18 pretrained on ImageNet
    - Frozen convolutional backbone
    - Trainable final fully connected layer

    This model is used for:
    - Baseline accuracy comparison
    - Grad-CAM explainability
    """

    def __init__(self, num_classes: int):
        super().__init__()

        # ----------------------------------------------------
        # Load pretrained ResNet-18
        # ----------------------------------------------------
        self.model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        # ----------------------------------------------------
        # Freeze ALL backbone layers
        # ----------------------------------------------------
        for param in self.model.parameters():
            param.requires_grad = False

        # ----------------------------------------------------
        # Replace classification head
        # ----------------------------------------------------
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: image tensor (B, 3, 224, 224)

        Returns:
            logits (B, num_classes)
        """
        return self.model(x)