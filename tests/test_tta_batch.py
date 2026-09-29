"""Batched inference (M10) must equal the per-image functions it replaces.

A small random CNN with BatchNorm stands in for the served model: the property
under test is "stacking views across images changes nothing", which holds for any
eval-mode network, so the 99 MB checkpoint is not needed.
"""
import numpy as np
import pytest
import torch
from PIL import Image

from calibration.temperature_scaling import TemperatureScaler
from inference.tta import single_predict, single_predict_batch, tta_predict, tta_predict_batch

DEVICE = torch.device("cpu")


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    net = torch.nn.Sequential(
        torch.nn.Conv2d(3, 6, 3, stride=4), torch.nn.BatchNorm2d(6), torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(6, 5))
    net.train()
    for _ in range(3):                      # give BatchNorm non-trivial running statistics
        net(torch.randn(16, 3, 224, 224))
    return net.eval()


@pytest.fixture(scope="module")
def scaler():
    s = TemperatureScaler()
    s.T = 0.6
    return s


def _images(n, seed=1):
    rng = np.random.default_rng(seed)
    sizes = [(300, 200), (64, 64), (500, 380), (240, 240), (96, 128)]
    return [Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)) for (w, h) in
            (sizes[i % len(sizes)] for i in range(n))]


def test_tta_batch_matches_per_image(model, scaler):
    imgs = _images(5)
    batched = tta_predict_batch(model, imgs, DEVICE, scaler=scaler)
    single = torch.stack([tta_predict(model, im, DEVICE, scaler=scaler) for im in imgs])
    assert batched.shape == (5, 5)
    assert torch.allclose(batched, single, atol=1e-5)


def test_single_batch_matches_per_image(model, scaler):
    imgs = _images(4, seed=2)
    batched = single_predict_batch(model, imgs, DEVICE, scaler=scaler)
    single = torch.stack([single_predict(model, im, DEVICE, scaler=scaler) for im in imgs])
    assert torch.allclose(batched, single, atol=1e-5)


def test_order_is_preserved(model, scaler):
    imgs = _images(3, seed=3)
    fwd = tta_predict_batch(model, imgs, DEVICE, scaler=scaler)
    rev = tta_predict_batch(model, imgs[::-1], DEVICE, scaler=scaler)
    assert torch.allclose(fwd, rev.flip(0), atol=1e-5)


def test_rows_are_probability_distributions(model, scaler):
    p = tta_predict_batch(model, _images(3), DEVICE, scaler=scaler)
    assert torch.allclose(p.sum(1), torch.ones(3), atol=1e-5)
    assert (p >= 0).all()


def test_without_scaler_uses_plain_softmax(model):
    imgs = _images(2)
    a = tta_predict_batch(model, imgs, DEVICE, scaler=None)
    b = torch.stack([tta_predict(model, im, DEVICE, scaler=None) for im in imgs])
    assert torch.allclose(a, b, atol=1e-5)


def test_empty_input_is_rejected(model):
    with pytest.raises(ValueError):
        tta_predict_batch(model, [], DEVICE)
    with pytest.raises(ValueError):
        single_predict_batch(model, [], DEVICE)


def test_chunking_does_not_change_results(model, scaler):
    imgs = _images(5, seed=4)
    whole = tta_predict_batch(model, imgs, DEVICE, scaler=scaler, max_images=8)
    chunked = tta_predict_batch(model, imgs, DEVICE, scaler=scaler, max_images=2)
    assert torch.allclose(whole, chunked, atol=1e-5)
