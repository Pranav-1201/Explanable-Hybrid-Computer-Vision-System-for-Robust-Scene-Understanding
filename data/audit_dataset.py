"""
data/audit_dataset.py

Audits the MIT Indoor 67 dataset for class balance and corrupt images.
Run once after any dataset or split change.
Outputs:
  - Console summary: min/max/mean images per class for train and test
  - results/train_class_dist.png — bar chart of train class distribution
  - results/test_class_dist.png  — bar chart of test class distribution
  - Console warning if any class has < 50 train images (triggers WeightedRandomSampler recommendation)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from data.dataset_loader import MITIndoorDataset

os.makedirs('results', exist_ok=True)

for split in ['train', 'test']:
    ds = MITIndoorDataset(f'data/MIT_Indoor/{split}', transform=None)
    labels = [ds.labels[i] for i in range(len(ds))]
    counts = np.bincount(labels, minlength=len(ds.classes))

    print(f"\n{split.upper()} split — {len(ds)} total images, {len(ds.classes)} classes")
    print(f"  min: {counts.min()} | max: {counts.max()} | mean: {counts.mean():.1f} | "
          f"std: {counts.std():.1f}")

    underrepresented = [ds.classes[i] for i, c in enumerate(counts) if c < 50]
    if underrepresented and split == 'train':
        print(f"\n  *** WARNING: {len(underrepresented)} classes with < 50 train images: ***")
        for cls in underrepresented:
            print(f"    {cls}: {counts[ds.classes.index(cls)]} images")
        print("  Recommendation: Add WeightedRandomSampler to train dataloader.")
    else:
        print(f"  All classes have >= 50 train images. No oversampling needed.")

    # Plot
    fig, ax = plt.subplots(figsize=(20, 5))
    ax.bar(range(len(counts)), sorted(counts, reverse=True))
    ax.axhline(50, color='red', linestyle='--', label='50-image threshold')
    ax.set_title(f'{split} class distribution (sorted descending)')
    ax.set_xlabel('Class rank')
    ax.set_ylabel('Image count')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'results/{split}_class_dist.png', dpi=150)
    plt.close()
    print(f"  Saved: results/{split}_class_dist.png")
