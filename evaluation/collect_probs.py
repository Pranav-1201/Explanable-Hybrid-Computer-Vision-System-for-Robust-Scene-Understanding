"""Collect the served model's calibrated TTA probabilities on three image sets.

    venv\\Scripts\\python.exe evaluation/collect_probs.py --field-root "<photo root>" --out <dir>

Writes <out>/probs_{val,test,field}.npz, each with `probs` (N, 67), `labels`
(N,; class index, -1 = out-of-distribution) and `names`. M2 (review routing)
sets its thresholds from `val` only; `test` and `field` are for reporting, so a
threshold is never tuned on the data it is scored on.

`val` is the same 10% split of the train folder (seed 42) that calibration
used -- the served checkpoint never trained on it.
"""
import argparse
import csv
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from calibration.temperature_scaling import TemperatureScaler  # noqa: E402
from inference.tta import tta_predict  # noqa: E402
from data.dataset_loader import MITIndoorDataset  # noqa: E402
from models.cnn_baseline import CNNBaseline  # noqa: E402
import serving.uploads  # noqa: E402,F401  (registers the HEIC/AVIF opener)
from utils.checkpoint import load_checkpoint  # noqa: E402

MODELS = os.path.join(ROOT, "models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL_SPLIT, SEED = 0.1, 42


def load_served_model(weights=None):
    classes = json.load(open(os.path.join(MODELS, "classes.json")))
    path = weights or os.path.join(MODELS, "phase2_ema.pth")
    meta = torch.load(path, map_location="cpu", weights_only=True)
    model = CNNBaseline(len(classes), backbone=meta["backbone"], pretrained=False)
    load_checkpoint(path, model, device=DEVICE)
    model.to(DEVICE).eval()
    scaler = TemperatureScaler.load(os.path.join(MODELS, "temperature_cnn.json"))
    return model, scaler, classes


def probs_for(model, scaler, paths):
    out = []
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            out.append(tta_predict(model, im.convert("RGB"), DEVICE, scaler=scaler).numpy())
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(paths)}", flush=True)
    return np.stack(out)


def folder_items(split_dir):
    """(path, label) in MITIndoorDataset order -- the order training and
    calibration used, which the seed-42 validation permutation depends on."""
    ds = MITIndoorDataset(root_dir=split_dir)
    return list(zip(ds.image_paths, ds.labels))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field-root", required=True, help="folder holding 'Testing dataset' and 'testing'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    model, scaler, classes = load_served_model(args.weights)

    def save(tag, items):
        paths = [p for p, _ in items]
        labels = np.array([y for _, y in items])
        print(f"{tag}: {len(paths)} images", flush=True)
        np.savez(os.path.join(args.out, f"probs_{tag}.npz"), probs=probs_for(model, scaler, paths),
                 labels=labels, names=np.array([os.path.basename(p) for p in paths]))

    train_items = folder_items(os.path.join(ROOT, "data", "MIT_Indoor", "train"))
    idx = torch.randperm(len(train_items), generator=torch.Generator().manual_seed(SEED)).tolist()
    save("val", [train_items[i] for i in idx[:int(len(train_items) * VAL_SPLIT)]])
    save("test", folder_items(os.path.join(ROOT, "data", "MIT_Indoor", "test")))

    cidx = {c: i for i, c in enumerate(classes)}
    labels_csv = os.path.join(ROOT, "evaluation", "field_labels.csv")
    field = []
    for r in csv.DictReader(open(labels_csv, encoding="utf-8")):
        folder = "Testing dataset" if r["group"] == "testing_dataset" else "testing"
        field.append((os.path.join(args.field_root, folder, r["filename"]),
                      cidx.get(r["true_class"], -1)))
    save("field", field)


if __name__ == "__main__":
    main()
