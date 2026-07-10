"""
Generate reliability diagram (calibration curve) comparing raw vs calibrated model.
Saves: results/reliability_diagram.png
"""
import sys, os, json
sys.path.insert(0, '.')
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — safe for servers
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from calibration.ece_metric import compute_ece

def plot_reliability_diagram(
    probs_raw:  np.ndarray,
    probs_cal:  np.ndarray,
    labels:     np.ndarray,
    save_path:  str = 'results/reliability_diagram.png',
    n_bins:     int = 15,
    model_name: str = 'Model',
):
    ece_raw, bins_raw = compute_ece(probs_raw, labels, n_bins, return_bins=True)
    ece_cal, bins_cal = compute_ece(probs_cal, labels, n_bins, return_bins=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Reliability Diagram — {model_name}', fontsize=14, fontweight='bold')

    for ax, bins, ece, title in [
        (axes[0], bins_raw, ece_raw, f'Before Calibration  (ECE={ece_raw*100:.1f}%)'),
        (axes[1], bins_cal, ece_cal, f'After Temperature Scaling  (ECE={ece_cal*100:.1f}%)'),
    ]:
        mid   = [(b['lo'] + b['hi']) / 2 for b in bins if b['n'] > 0]
        acc   = [b['acc']  for b in bins if b['n'] > 0]
        conf  = [b['conf'] for b in bins if b['n'] > 0]
        count = [b['n']    for b in bins if b['n'] > 0]

        ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfect calibration', alpha=0.6)
        bars = ax.bar(mid, acc, width=1/n_bins*0.85, alpha=0.7,
                      color='steelblue', label='Actual accuracy', align='center')
        ax.step([b['lo'] for b in bins if b['n'] > 0],
                conf, where='post', color='tomato', lw=2, label='Mean confidence')

        # Gap shading
        for m, a, c in zip(mid, acc, conf):
            lo, hi = min(a, c), max(a, c)
            color  = 'tomato' if c > a else 'green'
            ax.fill_between([m-0.5/n_bins, m+0.5/n_bins], lo, hi,
                            alpha=0.25, color=color)

        ax.set_xlabel('Confidence', fontsize=11)
        ax.set_ylabel('Accuracy',   fontsize=11)
        ax.set_title(title,         fontsize=11)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] Reliability diagram -> {save_path}")
    return ece_raw, ece_cal

if __name__ == '__main__':
    # Run this after calibration/run_calibration.py has been executed
    # and saved logits somewhere, or use the calibration report to
    # reconstruct the diagram from checkpoint
    print("To generate the reliability diagram, call plot_reliability_diagram()")
    print("with probs_raw and probs_cal arrays from the calibration run.")
    print("This is called automatically by run_calibration.py when run on GPU.")
