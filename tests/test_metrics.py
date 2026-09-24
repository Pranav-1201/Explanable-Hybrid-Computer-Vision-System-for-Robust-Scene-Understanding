import pytest

from serving.metrics import Metrics, percentile


def test_percentile_empty():
    assert percentile([], 50) is None


def test_percentile_single_value():
    assert percentile([42], 95) == 42


def test_percentile_known_values():
    values = list(range(1, 11))  # 1..10, already sorted
    assert percentile(values, 50) == pytest.approx(5.5)
    assert percentile(values, 95) == pytest.approx(9.55)
    assert percentile(values, 0) == 1
    assert percentile(values, 100) == 10


def test_metrics_snapshot_empty():
    m = Metrics()
    snap = m.snapshot()
    assert snap["tagged"] == 0
    assert snap["reviewed"] == 0
    assert snap["review_rate"] is None


def test_metrics_latency_series_independent():
    m = Metrics()
    m.record_latency("predict", 10.0)
    m.record_latency("predict", 20.0)
    m.record_latency("predict_batch", 500.0)
    snap = m.snapshot()
    assert snap["predict"]["requests"] == 2
    assert snap["predict_batch"]["requests"] == 1
    assert snap["predict"]["p50_ms"] == pytest.approx(15.0)


def test_metrics_review_rate():
    m = Metrics()
    for _ in range(3):
        m.record_outcome(in_scope=True)
    for _ in range(1):
        m.record_outcome(in_scope=False)
    snap = m.snapshot()
    assert snap["tagged"] == 3
    assert snap["reviewed"] == 1
    assert snap["review_rate"] == pytest.approx(0.25)


def test_metrics_bounded_by_max_samples():
    m = Metrics()
    for i in range(600):
        m.record_latency("predict", float(i))
    assert m.snapshot()["predict"]["requests"] == 500
