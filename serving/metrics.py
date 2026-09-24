"""In-process request metrics: latency percentiles and review-rate (B8).

Single-process only, same limitation as the in-memory rate limiter (B7) --
move both to a shared backend before running multiple replicas, or each
process reports its own disjoint numbers.
"""
import threading
from collections import deque

MAX_SAMPLES = 500


def percentile(sorted_values, pct):
    """Linear-interpolated percentile of an already-sorted sequence."""
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * (pct / 100)
    f, c = int(k), min(int(k) + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


class _LatencySeries:
    def __init__(self, max_samples=MAX_SAMPLES):
        self._values = deque(maxlen=max_samples)

    def record(self, ms):
        self._values.append(ms)

    def snapshot(self):
        values = sorted(self._values)
        return {
            "requests": len(values),
            "p50_ms": percentile(values, 50),
            "p95_ms": percentile(values, 95),
        }


class Metrics:
    """Thread-safe. One process-wide instance, held by app.py."""

    def __init__(self):
        self._lock = threading.Lock()
        self._series = {}
        self._tagged = 0
        self._review = 0

    def record_latency(self, series: str, ms: float):
        with self._lock:
            if series not in self._series:
                self._series[series] = _LatencySeries()
            self._series[series].record(ms)

    def record_outcome(self, in_scope: bool):
        with self._lock:
            if in_scope:
                self._tagged += 1
            else:
                self._review += 1

    def snapshot(self):
        with self._lock:
            series = {name: s.snapshot() for name, s in self._series.items()}
            tagged, review = self._tagged, self._review
        total = tagged + review
        return {
            **series,
            "tagged": tagged,
            "reviewed": review,
            "review_rate": (review / total) if total else None,
        }
