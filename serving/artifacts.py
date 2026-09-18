"""Serving artifacts: the class list and model weights.

The server must start from a fresh clone or container with no MIT Indoor
dataset on disk, so the class list is a committed JSON file and the weights
are a downloadable, checksummed artifact rather than whatever happens to sit
in models/.
"""
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request

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


def load_manifest(path: str) -> dict:
    """Return the weights entry of the serving manifest."""
    try:
        with open(path, encoding="utf-8") as f:
            weights = json.load(f)["weights"]
        entry = {"filename": str(weights["filename"]), "url": str(weights["url"]),
                 "sha256": str(weights["sha256"]).lower(), "size": int(weights["size"])}
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise ArtifactError(f"{path}: invalid serving manifest ({e})") from e
    return entry


def _verify(path: str, sha256: str, size: int) -> None:
    actual_size = os.path.getsize(path)
    if actual_size != size:
        raise ArtifactError(f"{path}: size {actual_size} bytes, expected {size}")
    actual = sha256_file(path)
    if actual != sha256.lower():
        raise ArtifactError(f"{path}: sha256 {actual}, expected {sha256}")


def ensure_weights(path: str, url: str, sha256: str, size: int,
                   fetch=urllib.request.urlopen) -> str:
    """Return `path` holding verified weights, downloading them if absent.

    An existing file that fails verification is an error, not a trigger to
    re-download: silently replacing a file an operator placed is worse than
    telling them. A download is written to a temp file in the same directory
    and renamed only after it verifies, so a partial file is never used.
    """
    if os.path.exists(path):
        try:
            _verify(path, sha256, size)
        except ArtifactError as e:
            raise ArtifactError(f"{e}. Delete it to re-download from {url}") from None
        return path

    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".part")
    try:
        try:
            with os.fdopen(fd, "wb") as out, fetch(url) as resp:
                shutil.copyfileobj(resp, out, _CHUNK)
        except (OSError, ValueError) as e:
            raise ArtifactError(f"download failed from {url}: {e}") from e
        _verify(tmp, sha256, size)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path
