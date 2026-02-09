# data/dataset_loader.py
# ============================================================
# MIT Indoor Scene Dataset Loader
# ------------------------------------------------------------
# - Loads images from class-wise folders
# - Used ONLY for CNN-based models
# - Clean, deterministic, and GPU-friendly
# ============================================================

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ------------------------------------------------------------
# Image Transform Pipelines
# ------------------------------------------------------------
def get_transforms(train: bool = True):
    """
    Returns torchvision transforms for training or evaluation.

    Train:
    - Resize
    - Random horizontal flip
    - Normalize (ImageNet stats)

    Test:
    - Resize
    - Normalize only
    """
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


# ------------------------------------------------------------
# Dataset Class
# ------------------------------------------------------------
class MITIndoorDataset(Dataset):
    """
    PyTorch Dataset for MIT Indoor Scenes.

    Directory structure expected:
    data/MIT_Indoor/train/<class_name>/*.jpg
    data/MIT_Indoor/test/<class_name>/*.jpg
    """

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_paths = []
        self.labels = []

        # Sorted class list for consistent label mapping
        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        # Build (image_path, label) pairs
        for label_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)

            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.image_paths.append(os.path.join(class_dir, fname))
                    self.labels.append(label_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Returns:
        - image tensor (3, 224, 224)
        - integer label
        """
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label