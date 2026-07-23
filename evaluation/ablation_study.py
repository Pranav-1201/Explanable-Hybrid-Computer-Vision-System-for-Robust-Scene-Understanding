"""evaluation/ablation_study.py — B-10 ablation table (inference only, no training).

Measures, on the held-out MIT Indoor 67 test set (n=1,340), what each component
of the system actually contributes:

  * +/- EMA   : phase-2 ResNet-50 raw weights vs the EMA weights we serve
  * +/- TTA   : served EMA model with and without test-time augmentation
  * +/- HOG   : full fusion head vs the same head with the HOG branch zeroed
                (HybridFusion.ablate_hog)

Methodological notes, stated up front because they change how the numbers read:

  1. The HOG row is a TEST-TIME ablation: the fusion head was TRAINED with real
     HOG features and we zero that branch at inference. It answers "how much is
     the trained head leaning on HOG?", NOT "how would a head trained without
     HOG perform". A retrained no-HOG control is a separate (training) job.
  2. Top-1/top-5 are invariant to temperature scaling (dividing logits by T > 0
     is monotonic), so calibration is deliberately not applied here; it would
     change confidences but not accuracy.
  3. The ResNet-18 row is CONTEXT, not a controlled Places365-vs-ImageNet
     ablation: it differs in both architecture and training recipe. A clean
     pretraining ablation needs an ImageNet ResNet-50 trained identically.

Run with the project venv from the repo root:
    venv\\Scripts\\python.exe evaluation/ablation_study.py
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader

from data.dataset_loader import MITIndoorDataset, get_transforms
from models.cnn_baseline import CNNBaseline
from models.hybrid_fusion import HybridFusion
from utils.checkpoint import load_checkpoint

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(ROOT, "models")
TEST_DIR = os.path.join(ROOT, "data", "MIT_Indoor", "test")
REPORT_PATH = os.path.join(ROOT, "reports", "ablation_table.md")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_CLASSES = 67


# ── metrics ─────────────────────────────────────────────────────────────────
def topk_hits(logits: torch.Tensor, labels: torch.Tensor):
    """Return (top1_hits, top5_hits) counts for a batch of logits."""
    top5 = logits.topk(5, dim=1).indices
    hit1 = (top5[:, 0] == labels).sum().item()
    hit5 = (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()
    return hit1, hit5


def as_pct(hit1, hit5, n):
    return 100.0 * hit1 / n, 100.0 * hit5 / n


# ── evaluators ──────────────────────────────────────────────────────────────
@torch.no_grad()
def eval_cnn(model) -> tuple:
    """Plain batched CNN evaluation over the test set."""
    ds = MITIndoorDataset(root_dir=TEST_DIR, transform=get_transforms(train=False))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    h1 = h5 = n = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        a, b = topk_hits(model(imgs), labels)
        h1 += a
        h5 += b
        n += labels.numel()
    return as_pct(h1, h5, n) + (n,)


@torch.no_grad()
def eval_cnn_tta(model) -> tuple:
    """Per-image TTA evaluation (inference/tta.tta_predict averages augmentations)."""
    from inference.tta import tta_predict

    ds = MITIndoorDataset(root_dir=TEST_DIR, transform=None)  # raw PIL
    h1 = h5 = 0
    n = len(ds)
    for i in range(n):
        pil, label = ds[i]
        probs = tta_predict(model, pil.convert("RGB"), DEVICE)  # (C,)
        a, b = topk_hits(probs.unsqueeze(0), torch.tensor([label]))
        h1 += a
        h5 += b
    return as_pct(h1, h5, n) + (n,)


@torch.no_grad()
def eval_fusion(cnn, fusion, ablate_hog: bool) -> tuple:
    """Fusion head over live images; optionally zero the HOG branch."""
    from training.train_fusion import ImageHogDataset

    ds = ImageHogDataset("test", get_transforms(train=False))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    h1 = h5 = n = 0
    for imgs, hogs, labels in loader:
        imgs, hogs, labels = imgs.to(DEVICE), hogs.to(DEVICE), labels.to(DEVICE)
        emb = cnn.get_embedding(imgs)
        logits = fusion.ablate_hog(emb) if ablate_hog else fusion(emb, hogs)
        a, b = topk_hits(logits, labels)
        h1 += a
        h5 += b
        n += labels.numel()
    return as_pct(h1, h5, n) + (n,)


# ── model builders ──────────────────────────────────────────────────────────
def build_phase2(load_ema: bool):
    path = os.path.join(MODELS_DIR, "phase2_best.pth")
    meta = torch.load(path, map_location="cpu", weights_only=True)
    backbone = meta.get("backbone", "resnet50_places365_local")
    model = CNNBaseline(NUM_CLASSES, backbone=backbone)
    load_checkpoint(path, model, device=DEVICE, load_ema=load_ema)
    return model


def build_fusion():
    path = os.path.join(MODELS_DIR, "fusion_best.pth")
    ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
    cnn = CNNBaseline(NUM_CLASSES, backbone=ckpt.get("backbone", "resnet50_places365_local"))
    cnn.load_state_dict(ckpt["cnn_state"])
    cnn.to(DEVICE).eval()
    fusion = HybridFusion(cnn_dim=ckpt.get("cnn_dim", 2048),
                          hog_dim=ckpt.get("hog_dim", 512),
                          num_classes=NUM_CLASSES)
    fusion.load_state_dict(ckpt["fusion_state"])
    fusion.to(DEVICE).eval()
    return cnn, fusion


def build_resnet18():
    from training.train_baseline import build_model
    model = build_model(NUM_CLASSES)
    load_checkpoint(os.path.join(MODELS_DIR, "baseline.pth"), model, device=DEVICE)
    return model


# ── main ────────────────────────────────────────────────────────────────────
def main():
    print(f"[INFO] device={DEVICE}")
    rows = []

    def record(name, note, fn):
        print(f"[RUN ] {name} ...", flush=True)
        top1, top5, n = fn()
        print(f"[DONE] {name}: top1={top1:.2f}%  top5={top5:.2f}%  (n={n})")
        rows.append((name, top1, top5, note))

    # --- CNN-only: +/- EMA, +/- TTA -----------------------------------------
    record("ResNet-50 Places365, raw weights", "no EMA, no TTA",
           lambda: eval_cnn(build_phase2(load_ema=False)))
    record("ResNet-50 Places365, EMA (SERVED)", "the deployed model",
           lambda: eval_cnn(build_phase2(load_ema=True)))
    record("ResNet-50 Places365, EMA + TTA", "test-time augmentation",
           lambda: eval_cnn_tta(build_phase2(load_ema=True)))

    # --- Fusion: +/- HOG ----------------------------------------------------
    if os.path.exists(os.path.join(MODELS_DIR, "fusion_best.pth")):
        cnn, fusion = build_fusion()
        record("Fusion (CNN + HOG), full", "disabled from serving",
               lambda: eval_fusion(cnn, fusion, ablate_hog=False))
        record("Fusion, HOG branch zeroed", "TEST-TIME ablation, not retrained",
               lambda: eval_fusion(cnn, fusion, ablate_hog=True))
    else:
        print("[SKIP] fusion_best.pth not found")

    # --- Context row --------------------------------------------------------
    if os.path.exists(os.path.join(MODELS_DIR, "baseline.pth")):
        record("ResNet-18 ImageNet baseline", "CONTEXT: different arch + recipe",
               lambda: eval_cnn(build_resnet18()))

    # --- Report -------------------------------------------------------------
    lines = [
        "# Ablation study (B-10)",
        "",
        "MIT Indoor 67 test set, n = 1,340. Inference only; no model was retrained.",
        "",
        "| Configuration | Test top-1 | Test top-5 | Note |",
        "|---|---|---|---|",
    ]
    for name, top1, top5, note in rows:
        lines.append(f"| {name} | {top1:.2f}% | {top5:.2f}% | {note} |")
    lines += [
        "",
        "## How to read this",
        "",
        "- **HOG row is a test-time ablation.** The fusion head was trained with real",
        "  HOG features; here that branch is zeroed at inference. It measures how much",
        "  the trained head leans on HOG, not how a head trained without HOG would do.",
        "- **Top-1/top-5 are invariant to temperature scaling** (dividing logits by",
        "  T > 0 is monotonic), so calibration is not applied; it moves confidences,",
        "  not accuracy.",
        "- **The ResNet-18 row is context, not a controlled ablation** - it differs in",
        "  architecture and training recipe, so it does not isolate Places365 vs",
        "  ImageNet pretraining. That control needs an identically-trained ImageNet",
        "  ResNet-50.",
        "",
    ]
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[SAVED] {REPORT_PATH}")


if __name__ == "__main__":
    main()
