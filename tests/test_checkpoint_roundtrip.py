"""Regression guard for utils.checkpoint.load_checkpoint (audit N4/N5/N11):
every checkpoint container this project emits must round-trip, EMA selection
must pick ema_state, the 'model.' prefix must remap in either direction, and an
architecture mismatch must fail loudly instead of loading garbage.
"""
import torch
import torch.nn as nn
import pytest

from utils.checkpoint import load_checkpoint


class Tiny(nn.Module):
    def __init__(self, d_in=4, d_out=3):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out)

    def forward(self, x):
        return self.fc(x)


class Wrapped(nn.Module):
    """Mimics CNNBaseline: the real weights live under a 'model.' attribute,
    so its state_dict keys are 'model.weight' / 'model.bias'."""
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(4, 3)

    def forward(self, x):
        return self.model(x)


def _sd_equal(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def _fresh():
    m = Tiny()
    for p in m.parameters():          # randomize so "loaded == source" is real
        nn.init.normal_(p)
    return m


def test_raw_state_dict_roundtrip(tmp_path):
    src = _fresh()
    p = tmp_path / "raw.pth"
    torch.save(src.state_dict(), p)
    dst = load_checkpoint(str(p), Tiny(), device="cpu")
    assert _sd_equal(src.state_dict(), dst.state_dict())


def test_state_dict_container_roundtrip(tmp_path):
    src = _fresh()
    p = tmp_path / "sd.pth"
    torch.save({"state_dict": src.state_dict()}, p)
    dst = load_checkpoint(str(p), Tiny(), device="cpu")
    assert _sd_equal(src.state_dict(), dst.state_dict())


def test_model_state_selected_when_load_ema_false(tmp_path):
    raw, ema = _fresh(), _fresh()
    p = tmp_path / "phase2.pth"
    torch.save({"model_state": raw.state_dict(), "ema_state": ema.state_dict()}, p)
    dst = load_checkpoint(str(p), Tiny(), device="cpu", load_ema=False)
    assert _sd_equal(raw.state_dict(), dst.state_dict())
    assert not _sd_equal(ema.state_dict(), dst.state_dict())


def test_ema_state_selected_when_load_ema_true(tmp_path):
    raw, ema = _fresh(), _fresh()
    p = tmp_path / "phase2.pth"
    torch.save({"model_state": raw.state_dict(), "ema_state": ema.state_dict()}, p)
    dst = load_checkpoint(str(p), Tiny(), device="cpu", load_ema=True)
    assert _sd_equal(ema.state_dict(), dst.state_dict())
    assert not _sd_equal(raw.state_dict(), dst.state_dict())


def test_model_prefix_remap(tmp_path):
    inner = nn.Linear(4, 3)
    p = tmp_path / "unwrapped.pth"
    torch.save(inner.state_dict(), p)                 # keys: weight, bias
    dst = load_checkpoint(str(p), Wrapped(), device="cpu")  # needs model.weight...
    assert torch.equal(dst.model.weight, inner.weight)
    assert torch.equal(dst.model.bias, inner.bias)


def test_architecture_mismatch_fails_loudly(tmp_path):
    p = tmp_path / "wrong.pth"
    torch.save(nn.Linear(999, 3).state_dict(), p)
    with pytest.raises(RuntimeError):
        load_checkpoint(str(p), Tiny(), device="cpu")


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(str(tmp_path / "nope.pth"), Tiny(), device="cpu")
