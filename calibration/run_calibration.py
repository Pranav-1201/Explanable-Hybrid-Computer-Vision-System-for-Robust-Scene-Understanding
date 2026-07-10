"""
Run post-training confidence calibration.
Loads phase2_best.pth, collects val logits, fits temperature T.
Saves: models/temperature_cnn.json
       results/calibration_report.json

Usage: python calibration/run_calibration.py
"""
import sys, os, json
sys.path.insert(0, '.')
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from data.dataset_loader import MITIndoorDataset, get_transforms
from models.cnn_baseline  import CNNBaseline
from utils.checkpoint     import load_checkpoint
from calibration.temperature_scaling import TemperatureScaler
from calibration.ece_metric          import compute_ece, calibration_summary

CHECKPOINT  = 'models/phase2_best.pth'
TRAIN_DIR   = 'data/MIT_Indoor/train'
VAL_SPLIT   = 0.1
SEED        = 42
BATCH_SIZE  = 128
NUM_CLASSES = 67

def collect_logits(model, loader, device):
    """Run model on loader, return (logits, labels) as numpy arrays."""
    model.eval()
    all_logits, all_labels = [], []
    use_amp = torch.cuda.is_available()
    from contextlib import nullcontext
    amp_ctx = torch.amp.autocast('cuda') if use_amp else nullcontext()

    with torch.no_grad(), amp_ctx:
        for imgs, lbls in loader:
            logits = model(imgs.to(device))
            all_logits.append(logits.cpu().numpy())
            all_labels.extend(lbls.numpy())

    return np.concatenate(all_logits), np.array(all_labels)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if not os.path.exists(CHECKPOINT):
        print(f"[ERROR] {CHECKPOINT} not found. Train the model first.")
        sys.exit(1)

    # Load model — use the backbone and the winning (raw vs EMA) weights that
    # training selected, so we calibrate exactly what will be served.
    meta     = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    backbone = meta.get('backbone', 'resnet50_places365_local')
    load_ema = bool(meta.get('best_is_ema', False))
    model = CNNBaseline(NUM_CLASSES, backbone=backbone).to(device)
    load_checkpoint(CHECKPOINT, model, device=device, load_ema=load_ema)
    model.eval()
    print(f"Loaded {CHECKPOINT}: backbone={backbone} load_ema={load_ema} "
          f"val_acc={meta.get('val_acc', float('nan')):.2f}%")

    # Build val split (same seed as training — CRITICAL for honest calibration)
    full_tmp = MITIndoorDataset(TRAIN_DIR, transform=None)
    n_total  = len(full_tmp)
    indices  = torch.randperm(n_total,
                   generator=torch.Generator().manual_seed(SEED)).tolist()
    n_val    = int(n_total * VAL_SPLIT)
    val_idx  = indices[:n_val]

    val_ds     = MITIndoorDataset(TRAIN_DIR, transform=get_transforms(train=False))
    val_subset = Subset(val_ds, val_idx)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, 
                            shuffle=False, num_workers=0)

    print(f"\nCollecting validation logits ({len(val_subset)} samples)...")
    logits, labels = collect_logits(model, val_loader, device)
    print(f"  Logits shape: {logits.shape}")

    # ECE before calibration
    probs_raw = np.exp(logits - logits.max(1, keepdims=True))
    probs_raw /= probs_raw.sum(1, keepdims=True)
    ece_raw   = compute_ece(probs_raw, labels)
    print(f"\nECE before calibration: {ece_raw*100:.2f}%")

    # Fit temperature scaling
    scaler = TemperatureScaler()
    scaler.fit(logits, labels)

    # ECE after calibration
    logits_cal = scaler.calibrate_logits(logits)
    probs_cal  = np.exp(logits_cal - logits_cal.max(1, keepdims=True))
    probs_cal /= probs_cal.sum(1, keepdims=True)

    ece_raw_final, ece_cal_final = calibration_summary(
        probs_raw, probs_cal, labels, 'ResNet-50 CNN'
    )

    # ── Rejection threshold from CALIBRATED confidence (fixes B-8) ──────────
    # The app previously hardcoded 0.30 against UNCALIBRATED confidence, which
    # over-rejected valid predictions on this underconfident model. Derive the
    # threshold from the calibrated max-prob of correctly-classified val images:
    # the 5th percentile accepts ~95% of confident-correct predictions while
    # still flagging genuinely low-confidence / OOD inputs.
    REJECT_PCTL = 5
    pred_cal    = probs_cal.argmax(1)
    conf_cal    = probs_cal.max(1)
    correct_msk = pred_cal == labels
    rejection_threshold = float(np.percentile(conf_cal[correct_msk], REJECT_PCTL))
    print(f"\nRejection threshold (calibrated {REJECT_PCTL}th pctl of correct preds): "
          f"{rejection_threshold:.4f}")
    print(f"  vs old hardcoded 0.30 against uncalibrated confidence")

    # Save
    scaler.save('models/temperature_cnn.json')
    with open('models/calibration_config.json', 'w') as f:
        json.dump({
            'temperature':         scaler.T,
            'rejection_threshold': rejection_threshold,
            'threshold_basis':     f'{REJECT_PCTL}th pctl of calibrated max-prob over correct val preds',
            'ece_before':          ece_raw_final,
            'ece_after':           ece_cal_final,
        }, f, indent=2)
    print(f"[SAVED] Calibration config -> models/calibration_config.json")

    report = {
        'model':      CHECKPOINT,
        'n_val':      len(val_subset),
        'temperature': scaler.T,
        'ece_before': ece_raw_final,
        'ece_after':  ece_cal_final,
        'ece_reduction_pct': (ece_raw_final - ece_cal_final) * 100,
        'target_met': ece_cal_final < 0.05,
    }
    os.makedirs('results', exist_ok=True)
    with open('results/calibration_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n[SAVED] Calibration report -> results/calibration_report.json")
    print(f"[SAVED] Temperature scalar -> models/temperature_cnn.json")

    from calibration.plot_reliability import plot_reliability_diagram
    plot_reliability_diagram(
        probs_raw, probs_cal, labels,
        save_path='results/reliability_diagram.png',
        model_name='ResNet-50 (MIT Indoor 67)',
    )
