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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ------------------------------------------------------------
# Image Transform Pipelines
# ------------------------------------------------------------
def get_transforms(train=False):
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
            transforms.Resize((256,256)),
            transforms.RandomResizedCrop(224),

            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),

            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05
            ),

            transforms.RandomPerspective(
                distortion_scale=0.2,
                p=0.3
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            )
        ])

    else:
        return transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),

            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
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

        # Use global class order from the training set
        base_dir = os.path.join(os.path.dirname(root_dir), "train")

        self.classes = sorted([
            d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))
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
    
# ------------------------------------------------------------
# DataLoader Helpers
# ------------------------------------------------------------
from torch.utils.data import DataLoader


def get_train_loader(batch_size=32, num_workers=4):
    """
    Returns DataLoader for training dataset.
    """

    dataset = MITIndoorDataset(
        root_dir="data/MIT_Indoor/train",
        transform=get_transforms(train=True)
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    return loader


def get_test_loader(batch_size=32, num_workers=4):
    """
    Returns DataLoader for test dataset.
    """

    dataset = MITIndoorDataset(
        root_dir="data/MIT_Indoor/test",
        transform=get_transforms(train=False)
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return loader