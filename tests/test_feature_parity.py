"""Regression guard for the train/serve feature skew (audit N8). ALL classical
feature extraction must go through the single source of truth
extract_features_from_rgb (RAW -> 128). If a caller re-introduces an
intermediate resize (e.g. the old serving path RAW -> 224 -> 128), features
drift and the tripwire below fails. This is the skew that is easiest to
silently reintroduce, so it is guarded most tightly.
"""
import numpy as np
from PIL import Image

from preprocessing.extract_hog_features import extract_features_from_rgb, compute_features

RNG = np.random.default_rng(0)


def _raw(h=300, w=400):
    return RNG.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def test_extractor_is_deterministic():
    img = _raw()
    assert np.array_equal(extract_features_from_rgb(img),
                          extract_features_from_rgb(img.copy()))


def test_compute_features_delegates_to_single_source():
    """The training path (compute_features) must be byte-identical to the shared
    extractor on the same RAW image — guards against a duplicate implementation
    diverging (exactly how the app.py copy drifted in N8)."""
    img = _raw()
    vec_train, label = compute_features((img, 7))
    vec_sot = extract_features_from_rgb(img)
    assert label == 7
    assert np.array_equal(vec_train, vec_sot)


def test_pre_resize_changes_features_regression_tripwire():
    """The N8 skew made concrete: feeding a 224-preresized image (the old
    serving path) yields materially different features than feeding RAW. This
    proves the resize entry point matters, so serving MUST pass the RAW image."""
    img = _raw()
    raw_feat = extract_features_from_rgb(img)
    pre224 = np.array(Image.fromarray(img).resize((224, 224), Image.LANCZOS))
    skew_feat = extract_features_from_rgb(pre224)
    assert np.abs(raw_feat - skew_feat).max() > 0.05   # documented skew ~0.27


def test_handles_grayscale_and_rgba_without_error():
    dim = extract_features_from_rgb(_raw()).shape[0]
    gray = RNG.integers(0, 256, size=(120, 160), dtype=np.uint8)
    rgba = RNG.integers(0, 256, size=(120, 160, 4), dtype=np.uint8)
    assert extract_features_from_rgb(gray).shape[0] == dim
    assert extract_features_from_rgb(rgba).shape[0] == dim
