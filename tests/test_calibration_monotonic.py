"""Regression guard for temperature scaling (audit B-6). Dividing logits by a
positive scalar T is monotonic: it must NEVER change the predicted class or the
per-row ranking, only the sharpness. Also checks fit stays in bounds / never
increases NLL and that save/load preserves T.
"""
import numpy as np
import torch
import pytest

from calibration.temperature_scaling import TemperatureScaler

RNG = np.random.default_rng(1)


@pytest.mark.parametrize("T", [0.2, 0.5, 1.0, 2.0, 5.0])
def test_temperature_preserves_argmax_and_ranking(T):
    logits = torch.tensor(RNG.normal(size=(50, 67)), dtype=torch.float32)
    s = TemperatureScaler()
    s.T = T
    probs = s.calibrate_probs(logits)
    assert torch.equal(probs.argmax(dim=1), logits.argmax(dim=1))
    assert torch.equal(probs.argsort(dim=1), logits.argsort(dim=1))


def test_calibrated_probs_are_valid_distribution():
    logits = torch.tensor(RNG.normal(size=(20, 10)), dtype=torch.float32)
    s = TemperatureScaler()
    s.T = 0.7
    probs = s.calibrate_probs(logits)
    assert torch.all(probs >= 0) and torch.all(probs <= 1)
    assert torch.allclose(probs.sum(dim=1), torch.ones(20), atol=1e-5)


def test_save_load_roundtrip_preserves_T(tmp_path):
    s = TemperatureScaler()
    s.T = 0.5142
    s.fitted = True
    p = tmp_path / "temperature.json"
    s.save(str(p))
    s2 = TemperatureScaler.load(str(p))
    assert s2.T == pytest.approx(0.5142)
    assert s2.fitted is True


def test_fit_returns_T_in_bounds_and_never_increases_nll():
    N, C = 200, 5
    labels = RNG.integers(0, C, size=N)
    logits = RNG.normal(size=(N, C))
    logits[np.arange(N), labels] += 2.0          # give the correct class signal
    s = TemperatureScaler()
    T = s.fit(logits, labels)
    assert 0.05 <= T <= 10.0
    assert s.fitted is True

    def nll(temp):
        sc = logits / temp
        sc = sc - sc.max(axis=1, keepdims=True)
        e = np.exp(sc)
        pr = e / e.sum(axis=1, keepdims=True)
        return -np.log(pr[np.arange(N), labels] + 1e-8).mean()

    assert nll(T) <= nll(1.0) + 1e-9              # fit can only help or tie
