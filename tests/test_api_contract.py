"""HTTP-level contract tests for the Flask API (T2, Phase C).

Imports app.py directly, which loads the real model at import time (module-
level `load_models()` call) -- paid once per test session via the session-
scoped `client` fixture, not once per test.
"""
import io

import numpy as np
import pytest
from PIL import Image


@pytest.fixture(scope="session")
def app_module():
    import app as app_module
    return app_module


@pytest.fixture(scope="session")
def client(app_module):
    app_module.app.testing = True
    with app_module.app.test_client() as c:
        yield c


def _image_bytes(fmt="JPEG", size=(64, 48)):
    buf = io.BytesIO()
    Image.new("RGB", size, (120, 80, 40)).save(buf, format=fmt)
    return buf.getvalue()


# ── /predict: oversize, wrong format, empty ─────────────────────

def test_predict_no_file_is_400(client):
    resp = client.post("/predict", data={}, content_type="multipart/form-data")
    assert resp.status_code == 400


def test_predict_empty_file_is_400(client):
    data = {"image": (io.BytesIO(b""), "x.jpg")}
    resp = client.post("/predict", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400


def test_predict_wrong_format_is_400(client):
    data = {"image": (io.BytesIO(_image_bytes("GIF")), "x.gif")}
    resp = client.post("/predict", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400


def test_predict_oversize_file_is_413(client, app_module):
    from serving.uploads import MAX_UPLOAD_BYTES
    oversize = b"\0" * (MAX_UPLOAD_BYTES + 1)
    data = {"image": (io.BytesIO(oversize), "big.jpg")}
    resp = client.post("/predict", data=data, content_type="multipart/form-data")
    assert resp.status_code == 413


def test_predict_valid_image_returns_200(client):
    data = {"image": (io.BytesIO(_image_bytes()), "x.jpg")}
    resp = client.post("/predict", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    body = resp.get_json()
    assert "prediction" in body and "confidence" in body


def test_predict_does_not_echo_original_image(client):
    """B6: /predict must not echo the upload back as base64 -- the client
    already has the bytes it sent."""
    data = {"image": (io.BytesIO(_image_bytes()), "x.jpg")}
    resp = client.post("/predict", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    assert "original_image" not in resp.get_json()


# ── /predict_batch: 51 files, per-file error isolation ──────────

def test_predict_batch_too_many_files_is_400(client, app_module):
    files = {"images": [(io.BytesIO(_image_bytes()), f"{i}.jpg")
                         for i in range(app_module.MAX_BATCH_IMAGES + 1)]}
    resp = client.post("/predict_batch", data=files, content_type="multipart/form-data")
    assert resp.status_code == 400


def test_predict_batch_exactly_max_files_is_accepted(client, app_module):
    files = {"images": [(io.BytesIO(_image_bytes()), f"{i}.jpg")
                         for i in range(app_module.MAX_BATCH_IMAGES)]}
    resp = client.post("/predict_batch", data=files, content_type="multipart/form-data")
    assert resp.status_code == 200
    assert len(resp.get_json()["results"]) == app_module.MAX_BATCH_IMAGES


def test_predict_batch_isolates_per_file_errors(client):
    files = {"images": [
        (io.BytesIO(_image_bytes()), "good.jpg"),
        (io.BytesIO(b""), "empty.jpg"),
    ]}
    resp = client.post("/predict_batch", data=files, content_type="multipart/form-data")
    assert resp.status_code == 200
    results = {r["filename"]: r for r in resp.get_json()["results"]}
    assert "error" not in results["good.jpg"]
    assert results["empty.jpg"]["review_reason"] == "invalid"


# ── threshold boundary (mocked, deterministic) ──────────────────

def _mock_single_predict_factory(app_module, target_class, target_prob):
    idx = app_module.classes.index(target_class)

    def _fake(model, pil_image, device, scaler=None):
        import torch
        probs = np.full(len(app_module.classes), 1e-6, dtype=np.float32)
        probs[idx] = target_prob
        return torch.from_numpy(probs)
    return _fake


@pytest.mark.parametrize("epsilon,expect_in_scope", [(1e-4, True), (-1e-4, False)])
def test_predict_batch_threshold_boundary(client, app_module, monkeypatch, epsilon, expect_in_scope):
    threshold = app_module.CONFIDENCE_THRESHOLD
    fake = _mock_single_predict_factory(app_module, "kitchen", threshold + epsilon)
    monkeypatch.setattr("inference.tta.single_predict", fake)

    files = {"images": [(io.BytesIO(_image_bytes()), "probe.jpg")]}
    resp = client.post("/predict_batch", data=files,
                        content_type="multipart/form-data", query_string={"tta": "0"})
    assert resp.status_code == 200
    row = resp.get_json()["results"][0]
    assert row["in_scope"] is expect_in_scope
    assert row["review_reason"] == (None if expect_in_scope else "low_confidence")


# ── B7: security headers + rate limiting wiring ─────────────────

def test_security_headers_present(client):
    resp = client.get("/health")
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["X-Frame-Options"] == "DENY"
    assert resp.headers["Content-Security-Policy"] == "default-src 'self'"


def test_predict_is_rate_limited(client):
    """Confirms the limiter is actually wired to /predict (not just imported):
    flask-limiter's per-request headers appear only on a limited route."""
    data = {"image": (io.BytesIO(_image_bytes()), "x.jpg")}
    resp = client.post("/predict", data=data, content_type="multipart/form-data")
    assert "X-RateLimit-Limit" in resp.headers


def test_health_is_not_rate_limited(client):
    """/health backs the Docker HEALTHCHECK -- it must never be throttled."""
    resp = client.get("/health")
    assert "X-RateLimit-Limit" not in resp.headers
