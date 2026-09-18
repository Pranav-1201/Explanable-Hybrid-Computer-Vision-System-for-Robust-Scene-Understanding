"""Serving artifacts: the class list must be loadable without the dataset."""
import hashlib
import io
import json
import os

import pytest

from serving.artifacts import ArtifactError, ensure_weights, load_classes, load_manifest, sha256_file

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _write(tmp_path, payload):
    p = tmp_path / "classes.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return str(p)


def test_load_classes_accepts_sorted_unique_names(tmp_path):
    assert load_classes(_write(tmp_path, ["bakery", "bar", "bedroom"])) == ["bakery", "bar", "bedroom"]


@pytest.mark.parametrize("payload", [[], {"a": 1}, ["b", "a"], ["a", "a"], ["a", ""], ["a", 3]])
def test_load_classes_rejects_malformed(tmp_path, payload):
    with pytest.raises(ArtifactError):
        load_classes(_write(tmp_path, payload))


def test_committed_classes_json_is_valid_67():
    classes = load_classes(os.path.join(ROOT, "models", "classes.json"))
    assert len(classes) == 67
    assert "bedroom" in classes and "kitchen" in classes


def test_sha256_file_matches_known_digest(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"abc")
    assert sha256_file(str(p)) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


PAYLOAD = b"weights" * 1000
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


def _fetcher(data):
    calls = []

    def fetch(url):
        calls.append(url)
        return io.BytesIO(data)

    return fetch, calls


def test_ensure_weights_downloads_and_verifies(tmp_path):
    target = tmp_path / "sub" / "w.pth"
    fetch, calls = _fetcher(PAYLOAD)
    assert ensure_weights(str(target), "http://x/w.pth", DIGEST, len(PAYLOAD), fetch=fetch) == str(target)
    assert target.read_bytes() == PAYLOAD and calls == ["http://x/w.pth"]


def test_ensure_weights_reuses_verified_file(tmp_path):
    target = tmp_path / "w.pth"
    target.write_bytes(PAYLOAD)
    fetch, calls = _fetcher(b"never")
    ensure_weights(str(target), "http://x/w.pth", DIGEST, len(PAYLOAD), fetch=fetch)
    assert calls == []


def test_tampered_download_is_rejected_and_not_kept(tmp_path):
    target = tmp_path / "w.pth"
    fetch, _ = _fetcher(PAYLOAD[:-1] + b"X")
    with pytest.raises(ArtifactError, match="sha256"):
        ensure_weights(str(target), "http://x/w.pth", DIGEST, len(PAYLOAD), fetch=fetch)
    assert list(tmp_path.iterdir()) == []


def test_corrupt_local_file_is_an_error_not_a_silent_redownload(tmp_path):
    target = tmp_path / "w.pth"
    target.write_bytes(b"short")
    fetch, calls = _fetcher(PAYLOAD)
    with pytest.raises(ArtifactError, match="size"):
        ensure_weights(str(target), "http://x/w.pth", DIGEST, len(PAYLOAD), fetch=fetch)
    assert calls == []


def test_download_failure_becomes_artifact_error(tmp_path):
    def fetch(url):
        raise OSError("connection refused")

    with pytest.raises(ArtifactError, match="download failed"):
        ensure_weights(str(tmp_path / "w.pth"), "http://x/w.pth", DIGEST, len(PAYLOAD), fetch=fetch)
    assert list(tmp_path.iterdir()) == []


def test_committed_manifest_is_complete():
    m = load_manifest(os.path.join(ROOT, "models", "serving_manifest.json"))
    assert m["filename"] == "phase2_ema.pth"
    assert m["url"].endswith("/releases/download/model-v1/phase2_ema.pth")
    assert len(m["sha256"]) == 64 and m["size"] > 0
