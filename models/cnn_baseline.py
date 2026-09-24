import torch
import torch.nn as nn
from torchvision import models

class CNNBaseline(nn.Module):
    """
    Baseline CNN for MIT Indoor Scene Classification with multiple backbone support.
    """

    def __init__(self, num_classes: int = 67, backbone: str = 'resnet50_places365_local',
                 pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone

        if not pretrained:
            # Architecture only: the caller loads a full checkpoint on top, so
            # reading or downloading pretrained weights here is wasted work and
            # a hard dependency on files/network a container does not have.
            if backbone.startswith('resnet50'):
                self.model = models.resnet50(weights=None)
            elif backbone == 'resnet18_imagenet':
                self.model = models.resnet18(weights=None)
            else:
                raise ValueError(f"Unknown backbone: {backbone}")
        elif backbone == 'resnet50_places365':
            try:
                self.model = torch.hub.load(
                    'CSAILVision/places365',
                    'resnet50_places365',
                    pretrained=True,
                    trust_repo=True
                )
            except Exception:
                print("[WARNING] Places365 hub load failed. Falling back to ResNet-50 ImageNet weights.")
                from torchvision.models import ResNet50_Weights
                self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        elif backbone == 'resnet50_places365_local':
            import os
            SAVE_PATH = 'models/resnet50_places365_weights.pth.tar'
            if os.path.exists(SAVE_PATH):
                self.model = models.resnet50(weights=None)
                self.model.fc = nn.Linear(self.model.fc.in_features, 365)
                checkpoint = torch.load(SAVE_PATH, map_location='cpu', weights_only=True)
                state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
                self.model.load_state_dict(state_dict)
            else:
                print(f"[WARNING] {SAVE_PATH} not found. Falling back to ResNet-50 ImageNet.")
                from torchvision.models import ResNet50_Weights
                self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        elif backbone == 'resnet50_imagenet':
            from torchvision.models import ResNet50_Weights
            self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        elif backbone == 'resnet18_imagenet':
            from torchvision.models import ResNet18_Weights
            self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        for param in self.model.parameters():
            param.requires_grad = True

        in_feats = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_feats, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def get_layer_groups(self):
        return {
            'layer1': list(self.model.layer1.parameters()),
            'layer2': list(self.model.layer2.parameters()),
            'layer3': list(self.model.layer3.parameters()),
            'layer4': list(self.model.layer4.parameters()),
            'head':   list(self.model.fc.parameters()),
        }

    def __repr__(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return f"CNNBaseline(backbone={self.backbone_name}, Total Params: {total:,}, Trainable: {trainable:,})"