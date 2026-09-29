"""Review routing (M2): decide whether a prediction is auto-tagged or sent to
the review queue, and why.

Rules run in a fixed order and the first that fires names the reason:

    low_confidence  calibrated top-1 probability below `conf_min`
    out_of_scope    the predicted class is not a home class
    low_margin      top-1 minus top-2 probability below `margin_min`  (optional)
    high_entropy    normalised predictive entropy above `entropy_max` (optional)

The two optional rules are OFF unless a threshold is supplied. Measured on the
held-out MIT test set they trade tag precision for home recall about one for
one (see reports/review_routing.md), so they are opt-in rather than default.
"""
import math
from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class RoutingPolicy:
    conf_min: float
    margin_min: Optional[float] = None
    entropy_max: Optional[float] = None

    def __post_init__(self):
        for name, lo, hi in (("conf_min", 0.0, 1.0), ("margin_min", 0.0, 1.0), ("entropy_max", 0.0, 1.0)):
            v = getattr(self, name)
            if v is not None and not lo <= v <= hi:
                raise ValueError(f"{name}={v} outside [{lo}, {hi}]")


def margin(probs: Sequence[float]) -> float:
    """Top-1 minus top-2 probability."""
    if len(probs) < 2:
        raise ValueError("margin needs at least two classes")
    top = sorted(probs, reverse=True)
    return float(top[0] - top[1])


def entropy(probs: Sequence[float]) -> float:
    """Shannon entropy divided by log(n): 0 = one-hot, 1 = uniform."""
    n = len(probs)
    if n < 2:
        raise ValueError("entropy needs at least two classes")
    h = -sum(p * math.log(p) for p in probs if p > 0.0)
    return float(h / math.log(n))


def review_reason(probs: Sequence[float], predicted_in_scope: bool, policy: RoutingPolicy) -> Optional[str]:
    """None when the photo can be auto-tagged, else the first rule that fired."""
    if max(probs) < policy.conf_min:
        return "low_confidence"
    if not predicted_in_scope:
        return "out_of_scope"
    if policy.margin_min is not None and margin(probs) < policy.margin_min:
        return "low_margin"
    if policy.entropy_max is not None and entropy(probs) > policy.entropy_max:
        return "high_entropy"
    return None
