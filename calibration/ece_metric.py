"""
Expected Calibration Error (ECE) and reliability diagram utilities.
ECE measures the gap between predicted confidence and actual accuracy.
A perfectly calibrated model has ECE = 0%.
Typical uncalibrated CNN: ECE 20-35%. Post-calibration target: ECE < 5%.
"""
import numpy as np


def compute_ece(
    probs:      np.ndarray,
    labels:     np.ndarray,
    n_bins:     int = 15,
    return_bins: bool = False,
):
    """
    Compute Expected Calibration Error.

    Args:
        probs:   (N, C) predicted probability arrays (after softmax)
        labels:  (N,) integer ground-truth labels
        n_bins:  Number of confidence bins (15 recommended)
        return_bins: If True, return per-bin stats for plotting

    Returns:
        ece:    scalar ECE value (lower is better, 0 = perfect)
        bins:   (optional) list of dicts with per-bin stats
    """
    assert probs.ndim == 2,   "probs must be (N, C)"
    assert labels.ndim == 1,  "labels must be (N,)"

    confidences = probs.max(axis=1)          # predicted confidence
    predictions = probs.argmax(axis=1)       # predicted class
    correct     = (predictions == labels).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece       = 0.0
    bins      = []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        n    = mask.sum()
        if n == 0:
            bins.append({'lo': lo, 'hi': hi, 'n': 0,
                         'acc': 0.0, 'conf': 0.0, 'gap': 0.0})
            continue
        acc_b  = correct[mask].mean()
        conf_b = confidences[mask].mean()
        gap    = abs(acc_b - conf_b)
        ece   += (n / len(labels)) * gap
        bins.append({
            'lo': lo, 'hi': hi, 'n': int(n),
            'acc': float(acc_b), 'conf': float(conf_b), 'gap': float(gap),
        })

    if return_bins:
        return float(ece), bins
    return float(ece)


def calibration_summary(
    probs_raw:   np.ndarray,
    probs_cal:   np.ndarray,
    labels:      np.ndarray,
    model_name:  str = 'Model',
    n_bins:      int = 15,
):
    """
    Print a calibration summary comparing raw vs calibrated model.
    """
    ece_raw = compute_ece(probs_raw, labels, n_bins)
    ece_cal = compute_ece(probs_cal, labels, n_bins)

    acc_raw = (probs_raw.argmax(1) == labels).mean()
    acc_cal = (probs_cal.argmax(1) == labels).mean()

    mean_conf_raw = probs_raw.max(1).mean()
    mean_conf_cal = probs_cal.max(1).mean()

    print(f"\n{'='*55}")
    print(f"  CALIBRATION SUMMARY — {model_name}")
    print(f"{'='*55}")
    print(f"  {'Metric':<28} {'Raw':>10} {'Calibrated':>12}")
    print(f"  {'-'*50}")
    print(f"  {'Accuracy':<28} {acc_raw*100:>9.2f}% {acc_cal*100:>11.2f}%")
    print(f"  {'Mean confidence':<28} {mean_conf_raw*100:>9.2f}% {mean_conf_cal*100:>11.2f}%")
    print(f"  {'ECE (lower is better)':<28} {ece_raw*100:>9.2f}% {ece_cal*100:>11.2f}%")
    print(f"  {'ECE improvement':<28} {'':>10} {(ece_raw-ece_cal)*100:>+11.2f}%")
    print(f"{'='*55}")

    target = 0.05
    status = '[PASS] TARGET MET' if ece_cal < target else f'[FAIL] TARGET MISSED (target: <{target*100:.0f}%)'
    print(f"  Calibration target (ECE < 5%): {status}")
    return ece_raw, ece_cal
