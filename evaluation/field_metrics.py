"""Metrics for the field evaluation (M5). Pure numpy: no model, no photos, so
the arithmetic is unit-testable and reusable by offline experiments.

Vocabulary (a photo has one true class, or -1 = matches none of the 67):

    auto-tagged   the routing policy accepts the prediction (confident + home class)
    right tag     auto-tagged AND predicted class == true class
    wrong tag     auto-tagged and not a right tag (includes any tag on a non-home photo)
    reviewed      everything not auto-tagged
    warranted     a reviewed photo whose auto-tag would have been wrong
"""
import math
from typing import Dict, Optional, Sequence

import numpy as np

from serving.routing import RoutingPolicy, review_reason


def wilson(k: int, n: int, z: float = 1.96):
    """95% Wilson score interval for k successes in n trials; (nan, nan) if n == 0."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def _ratio(k: int, n: int) -> Optional[float]:
    return None if n == 0 else k / n


def field_metrics(probs: np.ndarray, labels: np.ndarray, classes: Sequence[str],
                  home: Sequence[str], policy: RoutingPolicy) -> Dict:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    if probs.ndim != 2 or probs.shape[0] != labels.shape[0] or probs.shape[1] != len(classes):
        raise ValueError(f"shapes disagree: probs {probs.shape}, labels {labels.shape}, classes {len(classes)}")
    home_set = set(home)
    is_home_cls = np.array([c in home_set for c in classes])
    pred = probs.argmax(1)
    reasons = [review_reason(p, bool(is_home_cls[k]), policy) for p, k in zip(probs, pred)]
    tagged = np.array([r is None for r in reasons])
    labelled = labels >= 0
    right = tagged & (pred == labels)
    wrong = tagged & ~right
    true_home = np.zeros(len(labels), dtype=bool)
    true_home[labelled] = is_home_cls[labels[labelled]]

    n_tag, n_rev = int(tagged.sum()), int((~tagged).sum())
    warranted = int(((~tagged) & ~((pred == labels) & is_home_cls[pred])).sum())
    nonhome = ~true_home
    m = {
        "n": int(len(labels)),
        "n_labelled": int(labelled.sum()),
        "n_ood": int((~labelled).sum()),
        "n_true_home": int(true_home.sum()),
        "top1_labelled": _ratio(int((pred == labels)[labelled].sum()), int(labelled.sum())),
        "home_top1": _ratio(int((pred == labels)[true_home].sum()), int(true_home.sum())),
        "auto_tagged": n_tag,
        "right_tags": int(right.sum()),
        "wrong_tags": int(wrong.sum()),
        "tag_precision": _ratio(int(right.sum()), n_tag),
        "home_tag_recall": _ratio(int(right[true_home].sum()), int(true_home.sum())),
        "non_home_false_tags": int((tagged & nonhome).sum()),
        "non_home_photos": int(nonhome.sum()),
        "reviewed": n_rev,
        "review_rate": _ratio(n_rev, len(labels)),
        "warranted_reviews": warranted,
        "review_precision": _ratio(warranted, n_rev),
        "reasons": {r: reasons.count(r) for r in sorted({r for r in reasons if r})},
    }
    m["tag_precision_ci95"] = wilson(m["right_tags"], n_tag)
    m["home_tag_recall_ci95"] = wilson(int(right[true_home].sum()), int(true_home.sum()))
    return m
