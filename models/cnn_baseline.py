# models/cnn_baseline.py
# ============================================================
# Baseline CNN Model (ResNet-18)
# ============================================================

import torch
import torch.nn as nn
from torchvision import models


class CNNBaseline(nn.Module):
    """
    Baseline CNN for MIT Indoor Scene Classification.

    Architecture:
      - ResNet-18 pretrained on ImageNet
      - layer1 + layer2 frozen
      - layer3 + layer4 unfrozen (fine-tuned)
      - Improved two-stage classifier head:
          Dropout(0.4) → Linear(512→256) → BN → ReLU
          → Dropout(0.3) → Linear(256→num_classes)

    ⚠️  Head must match training/train_baseline.py build_model()
        exactly — mismatch causes load_state_dict() to fail.
    """

    def __init__(self, num_classes: int = 67):
        super().__init__()

        self.model = models.resnet18(pretrained=True)

        # ── Freeze backbone layers 1 & 2 ──────────────────────
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.layer3.parameters():
            param.requires_grad = True

        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # ── Two-stage classifier head ──────────────────────────
        in_features = self.model.fc.in_features  # 512 for ResNet-18

        self.model.fc = nn.Sequential(
            # Stage 1: 512 → 256
            nn.Dropout(0.4),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            # Stage 2: 256 → num_classes
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)