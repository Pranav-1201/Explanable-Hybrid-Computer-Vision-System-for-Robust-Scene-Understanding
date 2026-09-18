"""Export the serving artifacts from the Phase-2 training checkpoint.

Writes into dist/:
  phase2_ema.pth  - EMA weights only (no optimizer, scheduler or raw weights)
  classes.json    - sorted MIT Indoor class names, as training indexed them
  manifest.json   - sha256 and byte size of the weights

Then proves the slim checkpoint serves identical logits to the full one.
Run from the repo root: venv/Scripts/python.exe scripts/export_serving_artifacts.py
"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # CNNBaseline reads models/... relative to the working directory

import torch  # noqa: E402

from models.cnn_baseline import CNNBaseline  # noqa: E402
from serving.artifacts import sha256_file  # noqa: E402
from utils.checkpoint import load_checkpoint  # noqa: E402

SRC = os.path.join(ROOT, "models", "phase2_best.pth")
TRAIN_DIR = os.path.join(ROOT, "data", "MIT_Indoor", "train")
OUT = os.path.join(ROOT, "dist")


def main() -> None:
    ckpt = torch.load(SRC, map_location="cpu", weights_only=True)
    if not ckpt.get("best_is_ema"):
        raise SystemExit("phase2_best.pth has best_is_ema=False; serving expects EMA weights")
    state = ckpt["ema_state"]
    head = [k for k in state if k.endswith("fc.5.weight")]
    if len(head) != 1:
        raise SystemExit(f"expected exactly one classifier weight, found {head}")
    num_classes = int(state[head[0]].shape[0])

    classes = sorted(d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d)))
    if len(classes) != num_classes:
        raise SystemExit(f"{len(classes)} dataset classes but head has {num_classes} outputs")

    os.makedirs(OUT, exist_ok=True)
    weights_path = os.path.join(OUT, "phase2_ema.pth")
    torch.save({"state_dict": state, "backbone": ckpt["backbone"],
                "val_acc": float(ckpt["val_acc"]), "best_is_ema": True,
                "num_classes": num_classes}, weights_path)
    with open(os.path.join(OUT, "classes.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(classes, f, indent=2)
        f.write("\n")
    manifest = {"weights": {"filename": "phase2_ema.pth", "sha256": sha256_file(weights_path),
                            "size": os.path.getsize(weights_path)},
                "num_classes": num_classes}
    with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    full = CNNBaseline(num_classes, backbone=ckpt["backbone"])
    load_checkpoint(SRC, full, device="cpu", load_ema=True)
    slim = CNNBaseline(num_classes, backbone=ckpt["backbone"])
    load_checkpoint(weights_path, slim, device="cpu")
    torch.manual_seed(0)
    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        diff = (full(x) - slim(x)).abs().max().item()

    print(f"source   {SRC}  {os.path.getsize(SRC)} bytes")
    print(f"weights  {weights_path}  {manifest['weights']['size']} bytes")
    print(f"sha256   {manifest['weights']['sha256']}")
    print(f"classes  {num_classes}")
    print(f"parity   max|full-slim| logits = {diff}")
    if diff != 0.0:
        raise SystemExit("PARITY FAILED: slim checkpoint does not reproduce the served model")


if __name__ == "__main__":
    main()
