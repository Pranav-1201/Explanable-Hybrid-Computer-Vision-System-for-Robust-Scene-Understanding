import os
import shutil

RAW_DIR = "data/raw"
IMAGE_ROOT = os.path.join(RAW_DIR, "indoorCVPR_09", "Images")
TRAIN_LIST = os.path.join(RAW_DIR, "TrainImages.txt")
TEST_LIST = os.path.join(RAW_DIR, "TestImages.txt")

TARGET_ROOT = "data/MIT_Indoor"
TRAIN_DIR = os.path.join(TARGET_ROOT, "train")
TEST_DIR = os.path.join(TARGET_ROOT, "test")

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

def copy_images(list_file, target_base):
    with open(list_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()   # e.g. kitchen/img123.jpg
        class_name = line.split("/")[0]

        src_path = os.path.join(IMAGE_ROOT, line)
        dst_dir = os.path.join(target_base, class_name)
        os.makedirs(dst_dir, exist_ok=True)

        dst_path = os.path.join(dst_dir, os.path.basename(line))
        shutil.copy(src_path, dst_path)

print("Copying training images...")
copy_images(TRAIN_LIST, TRAIN_DIR)

print("Copying testing images...")
copy_images(TEST_LIST, TEST_DIR)

print("MIT Indoor dataset prepared successfully.")
