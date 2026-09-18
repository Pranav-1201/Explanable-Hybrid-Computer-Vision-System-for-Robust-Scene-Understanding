"""Serving artifacts: the class list must be loadable without the dataset."""
import json
import os

import pytest

from serving.artifacts import ArtifactError, load_classes, sha256_file

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
