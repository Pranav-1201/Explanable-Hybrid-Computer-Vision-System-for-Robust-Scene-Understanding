# training/train_fusion.py
"""
Architecture B — true feature fusion, two-stage training:
  Stage A: freeze the whole CNN, train the HybridFusion head on pre-extracted
           EMA embeddings + PCA-HOG features (fast, features from fusion_*.npz).
  Stage B: unfreeze the CNN's last block (layer4) and jointly fine-tune it with
           the fusion head, running images LIVE so gradients reach layer4
           (the frozen .npz embeddings cannot be used here — they are stale the
           moment layer4 changes).

Val split is carved from the TRAIN set with the seed-42 permutation shared by
train_baseline/train_phase2, so the fusion val set is exactly the CNN val set.
The test set is never used for any training decision — only a final report.
"""
import torch
import torch.nn as nn
import numpy as np
import os
import sys
import time
from contextlib import nullcontext
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Subset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataset_loader import MITIndoorDataset, get_transforms
from models.cnn_baseline import CNNBaseline
from models.hybrid_fusion import HybridFusion
from utils.checkpoint import load_checkpoint

PHASE2_CHECKPOINT = 'models/phase2_best.pth'
STAGE_A_EPOCHS    = 15
STAGE_B_EPOCHS    = 12
BATCH_SIZE_A      = 256    # features only — large batch is fine
BATCH_SIZE_B      = 32     # images through the CNN with layer4 grads — fits 8GB w/ AMP
BASE_LR_A         = 1e-3
BASE_LR_B_CNN     = 1e-5
BASE_LR_B_FUSION  = 1e-4
LABEL_SMOOTH      = 0.1
NUM_CLASSES       = 67
CNN_DIM           = 2048
HOG_DIM           = 512
SEED              = 42
VAL_SPLIT         = 0.1
MODEL_OUT         = 'models/fusion_best.pth'


class FusionDataset(torch.utils.data.Dataset):
    """Pre-extracted (cnn_embedding, hog_pca, label) for Stage A."""
    def __init__(self, split, data_dir='data'):
        data = np.load(os.path.join(data_dir, f'fusion_{split}.npz'))
        self.cnn    = torch.from_numpy(data['cnn']).float()
        self.hog    = torch.from_numpy(data['hog']).float()
        self.labels = torch.from_numpy(data['labels']).long()
    def __len__(self):  return len(self.labels)
    def __getitem__(self, i): return self.cnn[i], self.hog[i], self.labels[i]


class ImageHogDataset(torch.utils.data.Dataset):
    """(image_tensor, hog_pca, label) aligned by dataset index — for Stage B,
    where the CNN runs live so layer4 can be fine-tuned. Asserts the HOG rows
    are in the same dataset order as the images."""
    def __init__(self, split, transform):
        self.imgs = MITIndoorDataset(f'data/MIT_Indoor/{split}', transform=transform)
        hog = np.load(f'data/hog_pca_{split}.npz')
        self.hog = torch.from_numpy(hog['features']).float()
        assert len(self.imgs) == len(self.hog), "image/HOG count mismatch"
        assert np.array_equal(np.asarray(self.imgs.labels), hog['labels']), \
            "image/HOG label order mismatch — HOG features not in dataset order"
    def __len__(self):  return len(self.hog)
    def __getitem__(self, i):
        img, lbl = self.imgs[i]
        return img, self.hog[i], lbl


def split_train_val(n_total, val_split=VAL_SPLIT, seed=SEED):
    """Deterministic train/val index split over the fusion train set.

    Identical permutation (torch.randperm, seed 42, first 10% = val) to
    train_baseline/train_phase2, so the fusion val set is the exact same images
    the CNN validated on. The test set is never touched during training.
    """
    indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(seed)).tolist()
    n_val   = int(n_total * val_split)
    return indices[n_val:], indices[:n_val]


# ── Stage A: frozen features ────────────────────────────────────────────────
def run_epoch_a(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for cnn, hog, labels in loader:
            cnn, hog, labels = cnn.to(device), hog.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            out  = model(cnn, hog)
            loss = criterion(out, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            correct += (out.argmax(1) == labels).sum().item()
            total   += labels.size(0)
    return total_loss / len(loader), correct / total


# ── Stage B: images through the CNN (layer4 trainable) ──────────────────────
def run_epoch_b(cnn_model, fusion_model, loader, criterion, device,
                optimizer=None, scaler=None):
    train = optimizer is not None
    fusion_model.train(train)
    # Keep the CNN in eval() so the FROZEN layers' BatchNorm running stats stay
    # fixed; layer4's conv weights still learn (requires_grad=True + grad flow).
    # With lr=1e-5 over a short schedule this frozen-BN choice is stable.
    cnn_model.eval()
    amp = torch.cuda.is_available()
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for img, hog, labels in loader:
            img, hog, labels = img.to(device), hog.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            with (torch.amp.autocast('cuda') if amp else nullcontext()):
                emb  = cnn_model.get_embedding(img)     # grads reach layer4
                out  = fusion_model(emb, hog)
                loss = criterion(out, labels)
            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            total_loss += loss.item()
            correct += (out.argmax(1) == labels).sum().item()
            total   += labels.size(0)
    return total_loss / len(loader), correct / total


def save_fusion(path, cnn_model, fusion_model, backbone, stage, val_acc):
    torch.save({
        'stage':        stage,
        'cnn_state':    cnn_model.state_dict(),
        'fusion_state': fusion_model.state_dict(),
        'val_acc':      val_acc,
        'cnn_dim':      CNN_DIM,
        'hog_dim':      HOG_DIM,
        'backbone':     backbone,
    }, path)


if __name__ == '__main__':
    from utils.require_venv import require_project_venv
    require_project_venv(require_cuda=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(SEED)

    if os.environ.get('DRY_RUN') == '1':
        print("[DRY_RUN] imports/build only; skipping training.")
        sys.exit(0)

    for f in ('data/fusion_train.npz', 'data/hog_pca_train.npz'):
        if not os.path.exists(f):
            print(f"[ERROR] {f} not found. Run extract_cnn_embeddings.py, "
                  f"build_hog_pca.py, build_fusion_dataset.py first.")
            sys.exit(1)

    # Load the trained CNN with the WINNING (EMA) weights and recorded backbone.
    meta      = torch.load(PHASE2_CHECKPOINT, map_location='cpu', weights_only=True)
    backbone  = meta.get('backbone', 'resnet50_places365_local')
    load_ema  = bool(meta.get('best_is_ema', False))
    cnn_model = CNNBaseline(NUM_CLASSES, backbone=backbone).to(device)
    load_checkpoint(PHASE2_CHECKPOINT, cnn_model, device=device, load_ema=load_ema)
    for p in cnn_model.parameters():
        p.requires_grad = False
    cnn_model.eval()
    print(f"[cnn] {backbone}  load_ema={load_ema}  val_acc={meta.get('val_acc', float('nan')):.2f}%")

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    fusion_model = HybridFusion(cnn_dim=CNN_DIM, hog_dim=HOG_DIM, num_classes=NUM_CLASSES).to(device)

    # ── STAGE A: frozen CNN, features from fusion_train.npz ─────────────────
    full_a = FusionDataset('train')
    tr_idx, va_idx = split_train_val(len(full_a))
    a_train = DataLoader(Subset(full_a, tr_idx), batch_size=BATCH_SIZE_A, shuffle=True)
    a_val   = DataLoader(Subset(full_a, va_idx), batch_size=BATCH_SIZE_A, shuffle=False)
    print(f"Fusion split (no test leak): train={len(tr_idx)} val={len(va_idx)} (seed {SEED})")

    opt_a = torch.optim.AdamW(fusion_model.parameters(), lr=BASE_LR_A, weight_decay=1e-4)
    print("--- STAGE A (frozen CNN, fusion head) ---")
    best_val = 0.0
    for epoch in range(1, STAGE_A_EPOCHS + 1):
        t0 = time.time()
        tl, ta = run_epoch_a(fusion_model, a_train, criterion, device, optimizer=opt_a)
        vl, va = run_epoch_a(fusion_model, a_val,   criterion, device)
        flag = ""
        if va > best_val:
            best_val = va; flag = " <- best"
            save_fusion(MODEL_OUT, cnn_model, fusion_model, backbone, 'A', va)
        print(f"Stage A | Ep {epoch:2d}/{STAGE_A_EPOCHS} | loss {tl:.4f} | "
              f"train {ta*100:.1f}% | val {va*100:.1f}% | {time.time()-t0:.1f}s{flag}")
    stage_a_best = best_val
    print(f"[Stage A best val] {stage_a_best*100:.2f}%")

    # Resume Stage B from the best Stage-A fusion head.
    fusion_model.load_state_dict(torch.load(MODEL_OUT, weights_only=True)['fusion_state'])

    # ── STAGE B: unfreeze layer4, joint fine-tune on LIVE images ────────────
    for p in cnn_model.model.layer4.parameters():
        p.requires_grad = True

    b_train_ds = ImageHogDataset('train', get_transforms(train=True))
    b_val_ds   = ImageHogDataset('train', get_transforms(train=False))
    b_train = DataLoader(Subset(b_train_ds, tr_idx), batch_size=BATCH_SIZE_B,
                         shuffle=True, num_workers=0, pin_memory=True)
    b_val   = DataLoader(Subset(b_val_ds, va_idx), batch_size=BATCH_SIZE_B,
                         shuffle=False, num_workers=0, pin_memory=True)

    opt_b = torch.optim.AdamW([
        {'params': cnn_model.model.layer4.parameters(), 'lr': BASE_LR_B_CNN},
        {'params': fusion_model.parameters(),           'lr': BASE_LR_B_FUSION},
    ], weight_decay=1e-4)
    sched_b = CosineAnnealingLR(opt_b, T_max=STAGE_B_EPOCHS, eta_min=1e-6)
    scaler  = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    print("--- STAGE B (joint fine-tune layer4 + fusion) ---")
    for epoch in range(1, STAGE_B_EPOCHS + 1):
        t0 = time.time()
        tl, ta = run_epoch_b(cnn_model, fusion_model, b_train, criterion, device,
                             optimizer=opt_b, scaler=scaler)
        vl, va = run_epoch_b(cnn_model, fusion_model, b_val, criterion, device)
        sched_b.step()
        flag = ""
        if va > best_val:
            best_val = va; flag = " <- best"
            save_fusion(MODEL_OUT, cnn_model, fusion_model, backbone, 'B', va)
        print(f"Stage B | Ep {epoch:2d}/{STAGE_B_EPOCHS} | loss {tl:.4f} | "
              f"train {ta*100:.1f}% | val {va*100:.1f}% | {time.time()-t0:.1f}s{flag}")

    print(f"[Stage A best val] {stage_a_best*100:.2f}%   "
          f"[Overall best val] {best_val*100:.2f}%")

    # ── FINAL TEST (reporting only; never used for selection) ───────────────
    ckpt = torch.load(MODEL_OUT, map_location=device, weights_only=True)
    cnn_model.load_state_dict(ckpt['cnn_state'])
    fusion_model.load_state_dict(ckpt['fusion_state'])
    test_ds = ImageHogDataset('test', get_transforms(train=False))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE_B, shuffle=False,
                             num_workers=0, pin_memory=True)
    _, test_acc = run_epoch_b(cnn_model, fusion_model, test_loader, criterion, device)
    print(f"[FUSION TEST top1] {test_acc*100:.2f}%  (best stage={ckpt['stage']}, "
          f"val={ckpt['val_acc']*100:.2f}%)  | CNN-only ref: 83.21%")
