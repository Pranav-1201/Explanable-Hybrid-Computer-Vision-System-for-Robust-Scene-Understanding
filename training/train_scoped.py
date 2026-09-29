"""M3: scoped retrain -- 24 home classes + "other" (the 43 non-home classes).

Starts from the served phase-2 EMA weights and replaces only the last Linear
(512 -> 25), so the backbone keeps everything it learned on 67 classes while the
head is trained on the decision the product actually makes: "which home room is
this, or is it something else?". The 67-class model can only route a non-home
photo to review AFTER it has picked one of 67 rooms; this head can say "other".

    venv\\Scripts\\python.exe training/train_scoped.py [--epochs 15] [--out models/scoped_best.pth]

Uses the same seed-42 10% validation split as phase 2 and calibration, so the
held-out MIT test set stays untouched for the comparison.
"""
import argparse
import json
import os
import random
import sys
import time
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms.v2 import CutMix, MixUp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.dataset_loader import MITIndoorDataset, get_transforms  # noqa: E402
from models.cnn_baseline import CNNBaseline  # noqa: E402
from serving.scope import HOME_CLASS_LABELS, OTHER_CLASS  # noqa: E402
from utils.checkpoint import load_checkpoint  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_DIR = os.path.join(ROOT, "data", "MIT_Indoor", "train")
SEED, VAL_SPLIT, BATCH_SIZE, LABEL_SMOOTH = 42, 0.1, 64, 0.1


def scoped_classes(classes67):
    """24 home classes in classes.json order, then 'other'."""
    return [c for c in classes67 if c in HOME_CLASS_LABELS] + [OTHER_CLASS]


def label_map(classes67):
    """67-class index -> scoped index (unknown/non-home -> the 'other' index)."""
    scoped = scoped_classes(classes67)
    return [scoped.index(c) if c in HOME_CLASS_LABELS else len(scoped) - 1 for c in classes67]


class Scoped(Dataset):
    def __init__(self, base, mapping):
        self.base, self.mapping = base, mapping

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        return x, self.mapping[y]


def build(num_classes67, scoped_n, base_ckpt, device):
    model = CNNBaseline(num_classes67, backbone="resnet50_places365_local", pretrained=False)
    load_checkpoint(base_ckpt, model, device=device)
    fc = model.model.fc
    fc[-1] = nn.Linear(fc[-1].in_features, scoped_n)  # fresh head, everything below kept
    return model.to(device)


def evaluate(net, loader, device):
    net.eval()
    hit = n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            hit += (net(x).argmax(1) == y).sum().item()
            n += y.numel()
    return 100.0 * hit / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--base", default=os.path.join(ROOT, "models", "phase2_ema.pth"))
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "scoped_best.pth"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = torch.cuda.is_available()
    classes67 = json.load(open(os.path.join(ROOT, "models", "classes.json")))
    scoped = scoped_classes(classes67)
    mapping = label_map(classes67)
    n_scoped = len(scoped)
    print(f"scoped head: {n_scoped} classes ({n_scoped - 1} home + '{OTHER_CLASS}')", flush=True)

    full = MITIndoorDataset(root_dir=TRAIN_DIR)
    assert full.classes == classes67, "train folders and classes.json disagree on class order"
    idx = torch.randperm(len(full), generator=torch.Generator().manual_seed(SEED)).tolist()
    nval = int(len(full) * VAL_SPLIT)
    train_ds = Scoped(Subset(MITIndoorDataset(root_dir=TRAIN_DIR, transform=get_transforms(True)), idx[nval:]), mapping)
    val_ds = Scoped(Subset(MITIndoorDataset(root_dir=TRAIN_DIR, transform=get_transforms(False)), idx[:nval]), mapping)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=args.workers, pin_memory=True)

    mixup, cutmix = MixUp(alpha=0.4, num_classes=n_scoped), CutMix(alpha=1.0, num_classes=n_scoped)

    def mix(x, y):
        r = random.random()
        return mixup(x, y) if r < 0.3 else cutmix(x, y) if r < 0.6 else (x, y)

    model = build(len(classes67), n_scoped, args.base, device)
    from timm.utils import ModelEmaV2
    ema = ModelEmaV2(model, decay=0.99, device=device)  # ~100-step window: the run is only ~1,100 steps

    g = model.get_layer_groups()
    opt = torch.optim.AdamW([
        {"params": g["layer1"], "lr": args.lr * 0.01}, {"params": g["layer2"], "lr": args.lr * 0.01},
        {"params": g["layer3"], "lr": args.lr * 0.1}, {"params": g["layer4"], "lr": args.lr * 0.5},
        {"params": g["head"], "lr": args.lr * 3},
    ], weight_decay=1e-4)
    warm = 1
    sched = SequentialLR(opt, [LinearLR(opt, start_factor=0.1, total_iters=warm),
                               CosineAnnealingLR(opt, T_max=max(1, args.epochs - warm), eta_min=1e-6)], milestones=[warm])
    crit = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    scaler = torch.amp.GradScaler("cuda") if amp else None
    ctx = (lambda: torch.amp.autocast("cuda")) if amp else nullcontext

    best = -1.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            with ctx():
                x, y = mix(x, y)
                loss = crit(model(x), y)
            if amp:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            ema.update(model)
        sched.step()
        raw, ema_acc = evaluate(model, val_loader, device), evaluate(ema.module, val_loader, device)
        use_ema = ema_acc > raw
        score = max(raw, ema_acc)
        flag = ""
        if score > best:
            best = score
            flag = f" <- best ({'ema' if use_ema else 'raw'})"
            torch.save({
                "model_state": model.state_dict(), "ema_state": ema.module.state_dict(),
                "best_is_ema": use_ema, "backbone": "resnet50_places365_local",
                "num_classes": n_scoped, "class_names": scoped, "val_acc": score, "epoch": epoch,
            }, args.out)
        print(f"Epoch [{epoch:2d}/{args.epochs}] loss {loss.item():.3f} | val raw {raw:.1f}% | val EMA {ema_acc:.1f}% | {time.time() - t0:.0f}s{flag}", flush=True)
    print(f"best val {best:.2f}% -> {args.out}")


if __name__ == "__main__":
    from utils.require_venv import require_project_venv
    require_project_venv(require_cuda=True)
    main()
