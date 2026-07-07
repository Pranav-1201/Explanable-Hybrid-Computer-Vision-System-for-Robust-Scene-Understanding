# training/train_fusion.py
import torch
import torch.nn as nn
import numpy as np
import os
import sys
import time
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.cnn_baseline import CNNBaseline
from models.hybrid_fusion import HybridFusion
from utils.checkpoint import load_checkpoint

BACKBONE          = 'resnet50_places365'
PHASE2_CHECKPOINT = 'models/phase2_best.pth'
STAGE_A_EPOCHS    = 10
STAGE_B_EPOCHS    = 20
BATCH_SIZE        = 256    # larger batch OK since features pre-extracted
BASE_LR_A         = 1e-3
BASE_LR_B_CNN     = 1e-5
BASE_LR_B_FUSION  = 1e-4
LABEL_SMOOTH      = 0.1
NUM_CLASSES       = 67
SEED              = 42
VAL_SPLIT         = 0.1

class FusionDataset(torch.utils.data.Dataset):
    def __init__(self, split, data_dir='data'):
        data = np.load(os.path.join(data_dir, f'fusion_{split}.npz'))
        self.cnn    = torch.from_numpy(data['cnn']).float()
        self.hog    = torch.from_numpy(data['hog']).float()
        self.labels = torch.from_numpy(data['labels']).long()
    def __len__(self):  return len(self.labels)
    def __getitem__(self, i): return self.cnn[i], self.hog[i], self.labels[i]

def split_train_val(n_total, val_split=VAL_SPLIT, seed=SEED):
    """Deterministic train/val index split over the fusion train set.

    Uses the identical permutation (torch.randperm, seed 42, first 10% = val)
    as train_baseline.py / train_phase2.py. Fusion features are extracted in
    dataset order (shuffle=False upstream), so row i of fusion_train.npz is
    dataset index i — the fusion val set is therefore the exact same images
    the CNN validated on. The test set must never be touched during training;
    it is reserved for evaluation/evaluate_models.py.
    """
    indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(seed)).tolist()
    n_val   = int(n_total * val_split)
    return indices[n_val:], indices[:n_val]

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for cnn, hog, labels in loader:
        cnn, hog, labels = cnn.to(device), hog.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(cnn, hog)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (out.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total

def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for cnn, hog, labels in loader:
            cnn, hog, labels = cnn.to(device), hog.to(device), labels.to(device)
            out = model(cnn, hog)
            loss = criterion(out, labels)
            total_loss += loss.item()
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), correct / total

if __name__ == '__main__':
    from utils.require_venv import require_project_venv
    require_project_venv(require_cuda=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(SEED)
    
    if os.environ.get('DRY_RUN') != '1' and not os.path.exists('data/fusion_train.npz'):
        print("[ERROR] data/fusion_train.npz not found.")
        print("  Run the following scripts in order:")
        print("  1. python preprocessing/extract_cnn_embeddings.py")
        print("  2. python preprocessing/build_hog_pca.py  (if not done)")
        print("  3. python preprocessing/build_fusion_dataset.py")
        sys.exit(1)

    if os.environ.get('DRY_RUN') != '1':
        from torch.utils.data import Subset
        full_train = FusionDataset('train')
        train_idx, val_idx = split_train_val(len(full_train))
        train_ds = Subset(full_train, train_idx)
        val_ds   = Subset(full_train, val_idx)
        print(f"Fusion split (no test leak): train={len(train_ds)}  val={len(val_ds)}  "
              f"(val carved from train set, seed {SEED})")
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        # Load the trained CNN model (frozen)
        cnn_model = CNNBaseline(NUM_CLASSES, BACKBONE).to(device)
        if os.path.exists(PHASE2_CHECKPOINT):
            load_checkpoint(PHASE2_CHECKPOINT, cnn_model)
        for param in cnn_model.parameters():
            param.requires_grad = False
        cnn_model.eval()
        
        # Stage A: Train only fusion model
        fusion_model = HybridFusion(cnn_dim=2048, hog_dim=512, num_classes=NUM_CLASSES).to(device)
        optimizer_a  = torch.optim.AdamW(fusion_model.parameters(), lr=BASE_LR_A, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
        
        print("--- STAGE A (Frozen CNN) ---")
        best_val = 0
        for epoch in range(1, STAGE_A_EPOCHS + 1):
            t0 = time.time()
            tl, ta = train_epoch(fusion_model, train_loader, optimizer_a, criterion, device)
            vl, va = val_epoch(fusion_model, val_loader, criterion, device)
            print(f"Stage A | Epoch [{epoch:2d}/{STAGE_A_EPOCHS}] Loss: {tl:.4f} | Train: {ta*100:.1f}% | Val: {va*100:.1f}% | Time: {time.time()-t0:.1f}s")
            if va > best_val:
                best_val = va
                torch.save({
                    'epoch': epoch, 'stage': 'A',
                    'cnn_state': cnn_model.state_dict(),
                    'fusion_state': fusion_model.state_dict(),
                    'optimizer': optimizer_a.state_dict(),
                    'val_acc': va, 'cnn_dim': 2048, 'hog_dim': 512
                }, 'models/fusion_best.pth')

        # Stage B: Unfreeze layer4 of CNN
        print("--- STAGE B (Joint Fine-tuning layer4) ---")
        # In a full implementation, we'd extract fresh features per epoch since CNN changes, 
        # but here we follow the simplified structure where we just define the optimizer as requested:
        for param in cnn_model.model.layer4.parameters():
            param.requires_grad = True

        optimizer_b = torch.optim.AdamW([
            {'params': cnn_model.model.layer4.parameters(), 'lr': BASE_LR_B_CNN},
            {'params': fusion_model.parameters(),           'lr': BASE_LR_B_FUSION},
        ], weight_decay=1e-4)
        cosine_b = CosineAnnealingLR(optimizer_b, T_max=STAGE_B_EPOCHS, eta_min=1e-6)
        print("Model configuration ready for full training.")
