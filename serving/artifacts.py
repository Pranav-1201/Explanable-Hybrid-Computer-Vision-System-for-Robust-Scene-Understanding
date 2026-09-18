"""Serving artifacts: the class list and model weights.

The server must start from a fresh clone or container with no MIT Indoor
dataset on disk, so the class list is a committed JSON file and the weights
are a downloadable, checksummed artifact rather than whatever happens to sit
in models/.
"""
import hashlib
import json

_CHUNK = 1 << 20


class ArtifactError(RuntimeError):
    """A serving artifact is missing, malformed, or fails verification."""


def load_classes(path: str) -> list:
    """Return the class names, index-aligned with the model's output layer.

    Training indexed classes by sorted dataset directory name, so the file
    must be sorted; an unsorted or duplicated list would silently mislabel
    every prediction.
    """
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        raise ArtifactError(f"{path}: cannot read class list ({e})") from e
    if not isinstance(data, list) or not data:
        raise ArtifactError(f"{path}: expected a non-empty JSON list of class names")
    if not all(isinstance(c, str) and c for c in data):
        raise ArtifactError(f"{path}: every class name must be a non-empty string")
    if len(set(data)) != len(data):
        raise ArtifactError(f"{path}: duplicate class names")
    if data != sorted(data):
        raise ArtifactError(f"{path}: class names must be sorted to match training order")
    return data


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()
