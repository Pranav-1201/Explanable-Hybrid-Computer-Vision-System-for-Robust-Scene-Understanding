# models/hybrid_cnn.py
# ============================================================
# IMPROVED: Deeper MLP with better regularization
# ------------------------------------------------------------
# Changes vs original:
#   - Wider first layer (2048) to handle the larger feature dim
#   - Added residual-style skip connection at the bottleneck
#   - Increased dropout at the widest layer, reduced at deeper
#   - LeakyReLU instead of ReLU (avoids dead neurons)
#   - L2 weight decay applied in optimizer, not here
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Small residual block for the MLP.
    Helps gradient flow and reduces overfitting on shallow nets.
    """
    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))


class HybridCNN(nn.Module):
    """
    MLP classifier for classical feature vectors (HOG + color + LBP).

    Architecture:
        Input (feature_dim)
          → Linear → BN → LeakyReLU → Dropout
          → Linear → BN → LeakyReLU → Dropout
          → ResidualBlock
          → Linear → BN → LeakyReLU → Dropout
          → Linear → num_classes
    """

    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()

        self.classifier = nn.Sequential(
            # Stage 1: compress to 1024
            nn.Linear(feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.4),

            # Stage 2: 1024 → 512
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.35),

            # Stage 3: residual at 512
            ResidualBlock(512, dropout=0.3),

            # Stage 4: 512 → 256
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.25),

            # Output
            nn.Linear(256, num_classes)
        )

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.1,
                                        nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Ensure input is on the same device as model parameters
        x = x.to(self.classifier[0].weight.device)
        return self.classifier(x)