# models/hybrid_fusion.py
"""
Architecture B: True Feature Fusion Hybrid
- Branch A: CNN embedding (2048-d from ResNet-50)
- Branch B: HOG/LBP/colour descriptor (PCA-compressed, configurable dim)
- Fusion: L2-normalized concatenation → MLP classifier → 67 classes

This is a TRUE hybrid — a single jointly-trainable model combining deep
features with classical handcrafted descriptors.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridFusion(nn.Module):
    """
    Args:
        cnn_dim:     Dimension of CNN embedding (2048 for ResNet-50)
        hog_dim:     Dimension of PCA-compressed HOG features (256)
        num_classes: Number of output classes (67)
        dropout_1:   Dropout rate after first fusion layer (0.4)
        dropout_2:   Dropout rate after second fusion layer (0.3)
    """
    def __init__(
        self,
        cnn_dim:     int = 2048,
        hog_dim:     int = 512,
        num_classes: int = 67,
        dropout_1:   float = 0.4,
        dropout_2:   float = 0.3,
    ):
        super().__init__()
        self.cnn_dim = cnn_dim
        self.hog_dim = hog_dim
        fused_dim    = cnn_dim + hog_dim  # 2304

        # Per-branch batch normalisation before L2 normalisation
        self.cnn_bn = nn.BatchNorm1d(cnn_dim)
        self.hog_bn = nn.BatchNorm1d(hog_dim)

        # MLP classifier on fused representation
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(dropout_1),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout_2),
            nn.Linear(512, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, cnn_feat: torch.Tensor, hog_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cnn_feat: (B, cnn_dim) — raw CNN embedding from get_embedding()
            hog_feat: (B, hog_dim) — PCA-compressed HOG descriptor
        Returns:
            logits: (B, num_classes)
        """
        # Normalise each branch independently
        cnn = F.normalize(self.cnn_bn(cnn_feat), dim=1)
        hog = F.normalize(self.hog_bn(hog_feat), dim=1)

        # Concatenate and classify
        fused = torch.cat([cnn, hog], dim=1)           # (B, 2304)
        return self.classifier(fused)

    def get_fused_embedding(
        self,
        cnn_feat: torch.Tensor,
        hog_feat: torch.Tensor
    ) -> torch.Tensor:
        """Return the 512-d fused representation before the final linear layer.
        Used for visualisation and downstream analysis."""
        cnn   = F.normalize(self.cnn_bn(cnn_feat), dim=1)
        hog   = F.normalize(self.hog_bn(hog_feat), dim=1)
        fused = torch.cat([cnn, hog], dim=1)
        # Run through all but the last linear layer
        for layer in list(self.classifier.children())[:-1]:
            fused = layer(fused)
        return fused  # (B, 512)

    def ablate_hog(self, cnn_feat: torch.Tensor) -> torch.Tensor:
        """Zero-out HOG branch for ablation study. Feeds zeros as HOG input."""
        B = cnn_feat.shape[0]
        hog_zeros = torch.zeros(B, self.hog_dim, device=cnn_feat.device)
        return self.forward(cnn_feat, hog_zeros)

    def __repr__(self):
        total  = sum(p.numel() for p in self.parameters())
        train  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"HybridFusion(\n"
            f"  cnn_dim={self.cnn_dim}, hog_dim={self.hog_dim}\n"
            f"  fused_dim={self.cnn_dim+self.hog_dim}\n"
            f"  Total params: {total:,}  Trainable: {train:,}\n"
            f")"
        )
