import os
import sys
import time
import copy
import json
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.dataset_loader import MITIndoorDataset, get_transforms
from models.cnn_baseline import CNNBaseline
from torchvision.transforms.v2 import MixUp, CutMix
from contextlib import nullcontext

# ============================================================
# HYPERPARAMETERS
# ============================================================
BACKBONE      = 'resnet50_places365_local'
NUM_CLASSES   = 67
BATCH_SIZE    = 64
NUM_EPOCHS    = 45
BASE_LR       = 1e-3
WARMUP_EPOCHS = 3
LABEL_SMOOTH  = 0.1
VAL_SPLIT     = 0.1
SEED          = 42
TRAIN_DIR     = "data/MIT_Indoor/train"
MODEL_OUT     = "models/phase2_best.pth"

# ============================================================
# DATA MIXING
# ============================================================
mixup_fn  = MixUp(alpha=0.4, num_classes=NUM_CLASSES)
cutmix_fn = CutMix(alpha=1.0, num_classes=NUM_CLASSES)

def apply_mixup_cutmix(images, labels):
    r = random.random()
    if r < 0.3:
        return mixup_fn(images, labels)
    elif r < 0.6:
        return cutmix_fn(images, labels)
    else:
        return images, labels

def get_dataloaders(num_workers: int = 8):
    full_tmp  = MITIndoorDataset(root_dir=TRAIN_DIR, transform=None)
    n_total   = len(full_tmp)
    num_classes = len(full_tmp.classes)
    
    indices   = torch.randperm(n_total, generator=torch.Generator().manual_seed(SEED)).tolist()
    n_val     = int(n_total * VAL_SPLIT)
    train_idx = indices[n_val:]
    val_idx   = indices[:n_val]

    train_ds = MITIndoorDataset(root_dir=TRAIN_DIR, transform=get_transforms(train=True))
    val_ds   = MITIndoorDataset(root_dir=TRAIN_DIR, transform=get_transforms(train=False))

    train_dataset = Subset(train_ds, train_idx)
    val_dataset   = Subset(val_ds, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, num_classes, full_tmp.classes

# ============================================================
# OPTIMIZER
# ============================================================
def get_optimizer(model, base_lr=1e-3):
    groups = model.get_layer_groups()
    param_groups = [
        {'params': groups['layer1'], 'lr': base_lr * 0.01},
        {'params': groups['layer2'], 'lr': base_lr * 0.01},
        {'params': groups['layer3'], 'lr': base_lr * 0.1},
        {'params': groups['layer4'], 'lr': base_lr * 0.5},
        {'params': groups['head'],   'lr': base_lr},
    ]
    return torch.optim.AdamW(param_groups, weight_decay=1e-4)

# ============================================================
# TRAINING
# ============================================================
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    amp_ctx = lambda: torch.amp.autocast('cuda') if use_amp else nullcontext()

    train_loader, val_loader, num_classes, class_names = get_dataloaders()
    model = CNNBaseline(num_classes=num_classes, backbone=BACKBONE).to(device)

    try:
        from timm.utils import ModelEmaV2
        ema_model = ModelEmaV2(model, decay=0.9998, device=device)
        use_ema = True
    except Exception as e:
        print(f"[WARNING] EMA disabled: {e}")
        use_ema = False

    optimizer = get_optimizer(model, BASE_LR)
    
    warmup_sched = LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[WARMUP_EPOCHS])

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

    best_val_acc = 0.0
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    log_file = open('results/phase2_train.log', 'a')

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()

            with amp_ctx():
                images, labels = apply_mixup_cutmix(images, labels)
                outputs = model(images)
                loss = criterion(outputs, labels)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            if use_ema:
                ema_model.update(model)

            preds = torch.argmax(outputs, dim=1)
            running_loss += loss.item() * images.size(0)
            if labels.ndim > 1:
                labels_for_acc = torch.argmax(labels, dim=1)
            else:
                labels_for_acc = labels
            running_correct += (preds == labels_for_acc).sum().item()
            running_total += images.size(0)

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total * 100

        # VAL RAW
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += images.size(0)
        val_acc_raw = val_correct / val_total * 100

        # VAL EMA
        val_acc_ema = 0.0
        if use_ema:
            ema_model.module.eval()
            val_correct_ema = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    outputs = ema_model.module(images)
                    preds = torch.argmax(outputs, dim=1)
                    val_correct_ema += (preds == labels).sum().item()
            val_acc_ema = val_correct_ema / val_total * 100

        scheduler.step()
        
        current_lr = optimizer.param_groups[-1]['lr']

        target_val_acc = val_acc_ema if use_ema else val_acc_raw
        improved = ""
        if target_val_acc > best_val_acc:
            best_val_acc = target_val_acc
            improved = " <- best"
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'ema_state': ema_model.module.state_dict() if use_ema else None,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'val_acc': target_val_acc,
                'backbone': BACKBONE,
            }, MODEL_OUT)

        epoch_time = time.time() - epoch_start
        
        if use_ema:
            print(f"Epoch [{epoch:2d}/{NUM_EPOCHS}] Loss: {train_loss:.4f} | Train: {train_acc:.1f}% | Val(raw): {val_acc_raw:.1f}% | Val(EMA): {val_acc_ema:.1f}% | LR(head): {current_lr:.2e} | Time: {epoch_time:.0f}s{improved}")
            log_line = {"epoch": epoch, "loss": train_loss, "train_acc": train_acc/100, "val_acc_raw": val_acc_raw/100, "val_acc_ema": val_acc_ema/100, "lr_head": current_lr}
        else:
            print(f"Epoch [{epoch:2d}/{NUM_EPOCHS}] Loss: {train_loss:.4f} | Train: {train_acc:.1f}% | Val(raw): {val_acc_raw:.1f}% | LR(head): {current_lr:.2e} | Time: {epoch_time:.0f}s{improved}")
            log_line = {"epoch": epoch, "loss": train_loss, "train_acc": train_acc/100, "val_acc_raw": val_acc_raw/100, "lr_head": current_lr}

        log_file.write(json.dumps(log_line) + "\n")
        log_file.flush()

    log_file.close()

if __name__ == "__main__":
    if os.environ.get('DRY_RUN') != '1':
        train()
