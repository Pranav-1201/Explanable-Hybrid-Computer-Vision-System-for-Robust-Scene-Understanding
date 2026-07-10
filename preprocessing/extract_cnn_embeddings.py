# preprocessing/extract_cnn_embeddings.py
"""
Extract penultimate-layer CNN embeddings from ResNet-50 for all train/test images.
Used by both Architecture A (CNN-embedding SVM) and Architecture B (fusion MLP).
Saves: data/cnn_embeddings_train.npz, data/cnn_embeddings_test.npz
Each file contains: embeddings (N, 2048), labels (N,)
"""
import torch, numpy as np, os, argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from data.dataset_loader import MITIndoorDataset, get_transforms
from models.cnn_baseline import CNNBaseline
from utils.checkpoint import load_checkpoint

CHECKPOINT_PATH = 'models/phase2_best.pth'
FALLBACK_PATH   = 'models/baseline_best.pth'   # Phase 1 fallback
BATCH_SIZE      = 128
NUM_WORKERS     = 4 if torch.cuda.is_available() else 0

def extract(split, model, device, save_path):
    dataset = MITIndoorDataset(
        f'data/MIT_Indoor/{split}',
        transform=get_transforms(train=False)
    )
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )
    model.eval()
    embeddings, labels = [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc=f'Extracting {split} embeddings'):
            emb = model.get_embedding(imgs.to(device))   # (B, 2048)
            embeddings.append(emb.cpu().numpy())
            labels.extend(lbls.numpy())
    emb_arr = np.concatenate(embeddings, axis=0)
    lbl_arr = np.array(labels)
    np.savez(save_path, embeddings=emb_arr, labels=lbl_arr)
    print(f"[SAVED] {split}: {emb_arr.shape} -> {save_path}")
    return emb_arr, lbl_arr

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load best available checkpoint
    model = CNNBaseline(67)
    if os.path.exists(CHECKPOINT_PATH):
        load_checkpoint(CHECKPOINT_PATH, model)
        print(f"[LOADED] Phase 2 checkpoint: {CHECKPOINT_PATH}")
    elif os.path.exists(FALLBACK_PATH):
        load_checkpoint(FALLBACK_PATH, model)
        print(f"[LOADED] Phase 1 fallback checkpoint: {FALLBACK_PATH}")
    else:
        print("[WARNING] No trained checkpoint found. Using random weights.")
        print("         Embeddings will be meaningless. Train a model first.")
    model.to(device)

    os.makedirs('data', exist_ok=True)
    extract('train', model, device, 'data/cnn_embeddings_train.npz')
    extract('test',  model, device, 'data/cnn_embeddings_test.npz')
    print("\n[DONE] CNN embeddings extracted and saved.")
