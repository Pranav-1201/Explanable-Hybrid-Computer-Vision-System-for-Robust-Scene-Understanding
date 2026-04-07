# preprocessing/extract_hog_features.py
# ============================================================
# IMPROVED: Rich Multi-Feature Extraction
# ------------------------------------------------------------
# Features extracted per image:
#   1. HOG (spatial pyramid: full + 2x2 + 4x4 regions)
#   2. Color histogram (HSV, 16 bins per channel)
#   3. LBP texture descriptor
#   4. Color moments (mean, std per channel)
#
# Why this matters:
#   - Plain HOG on 128x128 grayscale = 1764 dims, loses color
#     and global layout. Test accuracy ~17-20% is expected.
#   - Spatial pyramid pooling encodes where features occur,
#     which is crucial for scene recognition.
#   - Color is highly discriminative for indoor scenes
#     (e.g. greenhouses vs. computer rooms).
# ============================================================

import os
import sys
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
from skimage.feature import hog, local_binary_pattern
from data.dataset_loader import MITIndoorDataset


# ============================================================
# CONFIGURATION
# ============================================================
IMG_SIZE       = (128, 128)
ORIENTATIONS   = 9
PPC            = (8, 8)       # pixels_per_cell — finer than before (was 16,16)
CPB            = (2, 2)       # cells_per_block
LBP_RADIUS     = 3
LBP_N_POINTS   = 8 * LBP_RADIUS
COLOR_BINS     = 16           # per HSV channel

TRAIN_DIR = "data/MIT_Indoor/train"
TEST_DIR  = "data/MIT_Indoor/test"
OUT_TRAIN = "data/hog_features_train.npz"
OUT_TEST  = "data/hog_features_test.npz"


# ============================================================
# SPATIAL PYRAMID HOG
# ============================================================
def hog_spatial_pyramid(gray_img):
    """
    Compute HOG at 3 scales (full image, 2x2 grid, 4x4 grid).
    This encodes WHERE edges occur in the image, not just what
    orientations exist globally — critical for scene recognition.
    """
    parts = []

    # Level 0: full image HOG
    feat = hog(
        gray_img,
        orientations=ORIENTATIONS,
        pixels_per_cell=PPC,
        cells_per_block=CPB,
        block_norm="L2-Hys"
    )
    parts.append(feat)

    # Level 1: 2x2 spatial grid
    h, w = gray_img.shape
    for row in range(2):
        for col in range(2):
            patch = gray_img[row*h//2:(row+1)*h//2,
                             col*w//2:(col+1)*w//2]
            patch_resized = cv2.resize(patch, (64, 64))
            feat = hog(
                patch_resized,
                orientations=ORIENTATIONS,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm="L2-Hys"
            )
            parts.append(feat)

    return np.concatenate(parts)


# ============================================================
# COLOR HISTOGRAM (HSV)
# ============================================================
def color_histogram_hsv(bgr_img):
    """
    HSV histogram — much more robust than RGB to lighting changes.
    16 bins each for H, S, V → 48-dim descriptor.
    Normalized to sum=1 for lighting invariance.
    """
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    hist_parts = []
    for i, bins in enumerate([COLOR_BINS, COLOR_BINS, COLOR_BINS]):
        ranges = [(0, 180), (0, 256), (0, 256)][i]
        h = cv2.calcHist([hsv], [i], None, [bins], list(ranges))
        h = h.flatten().astype(np.float32)
        h /= (h.sum() + 1e-6)
        hist_parts.append(h)
    return np.concatenate(hist_parts)


# ============================================================
# COLOR MOMENTS
# ============================================================
def color_moments(bgr_img):
    """
    Mean and std of each BGR channel — compact 6-dim color summary.
    """
    moments = []
    for c in range(3):
        ch = bgr_img[:, :, c].astype(np.float32) / 255.0
        moments.append(ch.mean())
        moments.append(ch.std())
    return np.array(moments, dtype=np.float32)


# ============================================================
# LBP TEXTURE
# ============================================================
def lbp_histogram(gray_img):
    """
    Local Binary Pattern histogram — captures micro-texture.
    Important for distinguishing e.g. tiled bathroom vs carpet
    flooring. Uniform LBP has n_points+2 bins.
    """
    lbp = local_binary_pattern(
        gray_img,
        P=LBP_N_POINTS,
        R=LBP_RADIUS,
        method="uniform"
    )
    n_bins = LBP_N_POINTS + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins,
                           range=(0, n_bins), density=True)
    return hist.astype(np.float32)


# ============================================================
# MASTER FEATURE EXTRACTOR
# ============================================================
def compute_features(sample):
    """
    Extract ALL features from one (image, label) sample.
    Returns: (feature_vector, label)

    Expected total dims:
      - HOG pyramid:     ~3564  (full + 4 patches)
      - Color histogram: 48
      - Color moments:   6
      - LBP:             26
      --------------------------------
      Total:             ~3644
    """
    image, label = sample

    # Ensure numpy RGB
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    # Ensure 3-channel (some images may be RGBA or grayscale)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = image[:, :, :3]

    # Resize
    image_resized = cv2.resize(image, IMG_SIZE)

    # BGR for OpenCV color ops
    bgr = cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR)

    # Grayscale for HOG / LBP
    gray = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    # Build feature vector
    hog_feat   = hog_spatial_pyramid(gray)
    color_hist = color_histogram_hsv(bgr)
    c_moments  = color_moments(bgr)
    lbp_feat   = lbp_histogram(gray)

    feature_vec = np.concatenate([hog_feat, color_hist, c_moments, lbp_feat])
    return feature_vec.astype(np.float32), label


# ============================================================
# MAIN EXTRACTION
# ============================================================
def extract(split: str):
    assert split in {"train", "test"}

    out_path = OUT_TRAIN if split == "train" else OUT_TEST
    data_dir = TRAIN_DIR if split == "train" else TEST_DIR

    if os.path.exists(out_path):
        print(f"[SKIP] Features already exist: {out_path}")
        return

    print(f"\nExtracting rich features for: {split.upper()}")

    # Load WITHOUT transforms so we get raw PIL images
    dataset = MITIndoorDataset(root_dir=data_dir, transform=None)

    # IMPORTANT: do NOT shuffle — keep deterministic order
    # so features[i] always matches labels[i]
    samples = [dataset[i] for i in range(len(dataset))]

    results = []
    for sample in tqdm(samples, desc=f"Features ({split})"):
        results.append(compute_features(sample))

    features, labels = zip(*results)
    features = np.stack(features)
    labels   = np.array(labels, dtype=np.int64)

    os.makedirs("data", exist_ok=True)
    np.savez(out_path, features=features, labels=labels)

    print(f"[DONE] {split} features saved:")
    print(f"       Shape: {features.shape}")
    print(f"       Path:  {out_path}")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    # Always delete old feature files if re-running
    for p in [OUT_TRAIN, OUT_TEST, "data/hog_feature_stats.npz"]:
        if os.path.exists(p):
            os.remove(p)
            print(f"[CLEAN] Removed old file: {p}")

    extract("train")
    extract("test")

    print("\n[SUMMARY] Feature dimensions:")
    data = np.load(OUT_TRAIN)
    print(f"  Train: {data['features'].shape}")
    data = np.load(OUT_TEST)
    print(f"  Test:  {data['features'].shape}")