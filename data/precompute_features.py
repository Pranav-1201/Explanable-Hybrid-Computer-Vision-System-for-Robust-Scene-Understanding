import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
import cv2
import numpy as np
from tqdm import tqdm

from classical_features.feature_stack import stack_features

SETS = ["train", "test"]

for split in SETS:
    SRC_DIR = f"data/MIT_Indoor/{split}"
    DST_DIR = f"data/MIT_Indoor/features/{split}"

    os.makedirs(DST_DIR, exist_ok=True)

    for cls in os.listdir(SRC_DIR):
        src_cls = os.path.join(SRC_DIR, cls)
        dst_cls = os.path.join(DST_DIR, cls)
        os.makedirs(dst_cls, exist_ok=True)

        for img_name in tqdm(os.listdir(src_cls), desc=f"{split}/{cls}"):
            img_path = os.path.join(src_cls, img_name)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            features = stack_features(image)
            np.save(
                os.path.join(dst_cls, img_name.replace(".jpg", ".npy")),
                features.numpy()
            )


os.makedirs(DST_DIR, exist_ok=True)

for cls in os.listdir(SRC_DIR):
    src_cls = os.path.join(SRC_DIR, cls)
    dst_cls = os.path.join(DST_DIR, cls)
    os.makedirs(dst_cls, exist_ok=True)

    for img_name in tqdm(os.listdir(src_cls), desc=f"Processing {cls}"):
        img_path = os.path.join(src_cls, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        features = stack_features(image)
        np.save(os.path.join(dst_cls, img_name.replace(".jpg", ".npy")), features.numpy())
