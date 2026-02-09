import cv2
import numpy as np
import torch

from classical_features.edges import canny_edges
from classical_features.texture import hog_features, gabor_features



def normalize_feature(feature):
    """Normalize feature map to [0, 1]"""
    feature = feature.astype(np.float32)
    feature -= feature.min()
    feature /= (feature.max() + 1e-6)
    return feature


def stack_features(rgb_image, size=(224, 224)):
    """
    Input: RGB image (H, W, 3)
    Output: Stacked tensor (C, H, W) with fixed size
    """

    # Resize image FIRST (CRITICAL)
    rgb_image = cv2.resize(rgb_image, size)

    # Convert to grayscale
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

    # Classical features
    edges = canny_edges(gray)
    hog = hog_features(gray)
    gabor = gabor_features(gray)

    # Normalize all feature maps
    edges = normalize_feature(edges) * 1.2
    hog = normalize_feature(hog) * 1.1
    gabor = normalize_feature(gabor) * 1.3


    # Normalize RGB
    rgb = rgb_image.astype(np.float32) / 255.0

    # Stack channels
    stacked = np.dstack([
        rgb,
        edges[..., np.newaxis],
        hog[..., np.newaxis],
        gabor[..., np.newaxis]
    ])

    # Convert to PyTorch tensor (C, H, W)
    stacked_tensor = torch.tensor(stacked).permute(2, 0, 1)

    return stacked_tensor
