# preprocessing/extract_hog_features.py
# ============================================================
# HOG Feature Extraction (Optimized & Cached)
# ------------------------------------------------------------
# - Extracts HOG features ONCE for train & test sets
# - Uses CPU multiprocessing (i7-13650HX friendly)
# - Saves features as .npz for fast reuse
# - Absolutely NO recomputation during training
# ============================================================

import os
import sys
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.feature import hog
from data.dataset_loader import MITIndoorDataset


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
IMG_SIZE = (128, 128)              # Balance speed & accuracy
ORIENTATIONS = 9
PIXELS_PER_CELL = (16, 16)
CELLS_PER_BLOCK = (2, 2)

TRAIN_DIR = "data/MIT_Indoor/train"
TEST_DIR  = "data/MIT_Indoor/test"

OUT_TRAIN = "data/hog_features_train.npz"
OUT_TEST  = "data/hog_features_test.npz"


# ------------------------------------------------------------
# HOG COMPUTATION FUNCTION (WORKER SAFE)
# ------------------------------------------------------------
def compute_hog(sample):
    """
    Computes HOG features for a single image.

    Args:
        sample: (image_tensor, label)
    Returns:
        (hog_vector, label)
    """
    image, label = sample

    # Ensure image is NumPy (handles PIL safely)
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    # Convert to grayscale
    gray = rgb2gray(image)

    # Resize for consistency
    gray = resize(gray, IMG_SIZE, anti_aliasing=True)

    # Compute HOG descriptor
    hog_vec = hog(
        gray,
        orientations=ORIENTATIONS,
        pixels_per_cell=PIXELS_PER_CELL,
        cells_per_block=CELLS_PER_BLOCK,
        block_norm="L2-Hys"
    )

    return hog_vec.astype(np.float32), label


# ------------------------------------------------------------
# MAIN EXTRACTION FUNCTION
# ------------------------------------------------------------
def extract(split: str):
    assert split in {"train", "test"}

    out_path = OUT_TRAIN if split == "train" else OUT_TEST
    data_dir = TRAIN_DIR if split == "train" else TEST_DIR

    # --------------------------------------------------------
    # Skip if already extracted
    # --------------------------------------------------------
    if os.path.exists(out_path):
        print(f"[SKIP] HOG features already exist: {out_path}")
        return

    print(f"\nExtracting HOG features for: {split.upper()}")

    # --------------------------------------------------------
    # Load dataset WITHOUT transforms
    # --------------------------------------------------------
    dataset = MITIndoorDataset(
        root_dir=data_dir,
        transform=None
    )

    samples = [dataset[i] for i in range(len(dataset))]

    # --------------------------------------------------------
    # Multiprocessing on CPU
    # --------------------------------------------------------
    num_workers = max(1, cpu_count() - 2)

    with Pool(num_workers) as pool:
        results = list(
            tqdm(
                pool.imap(compute_hog, samples),
                total=len(samples),
                desc=f"HOG ({split})"
            )
        )

    features, labels = zip(*results)

    features = np.stack(features)
    labels = np.array(labels, dtype=np.int64)

    # --------------------------------------------------------
    # Save to disk
    # --------------------------------------------------------
    os.makedirs("data", exist_ok=True)
    np.savez(out_path, features=features, labels=labels)

    print(f"[DONE] Saved {split} HOG features:")
    print(f"       Features shape: {features.shape}")
    print(f"       Path: {out_path}")


# ------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------
if __name__ == "__main__":
    extract("train")
    extract("test")