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
import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_transforms(train=False):
    if train:
        return T.Compose([
            T.Resize((256, 256)),
            T.RandomResizedCrop(224, scale=(0.7, 1.0)),   # wider scale range vs old (0.08, 1.0)
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.05),                 # rare but helps scene layouts
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08),
            T.RandomGrayscale(p=0.05),
            T.RandomPerspective(distortion_scale=0.25, p=0.3),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            T.RandomErasing(p=0.3, scale=(0.02, 0.15)),   # MUST be after ToTensor — operates on tensors
        ])
    else:  # eval / test
        return T.Compose([
            T.Resize((232, 232)),    # slightly oversized for center crop — standard modern eval protocol
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
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