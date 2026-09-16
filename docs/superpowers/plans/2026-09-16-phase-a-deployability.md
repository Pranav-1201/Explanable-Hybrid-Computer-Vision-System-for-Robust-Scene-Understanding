# Phase A — Deployability & Correctness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the interior tagger run from a fresh clone or a CPU container with no dataset and no pre-placed weights, accept real phone photos (AVIF/HEIC) in 50-photo batches, and prove it with a GitHub Actions smoke job.

**Architecture:** A new `serving/` package owns everything the server needs that is not a Flask route: the class list and model-weights artifacts (`serving/artifacts.py`), upload validation (`serving/uploads.py`), and environment parsing (`serving/config.py`). `app.py` keeps the routes and wiring. Weights are a slim EMA-only checkpoint published as a GitHub Release asset and fetched at startup with a sha256 check. `serve.py` runs the app under waitress; a multi-stage Dockerfile builds `serve` and `test` targets that CI exercises.

**Tech Stack:** Python 3.10, Flask 3.1.3, PyTorch 2.5.1 / torchvision 0.20.1 (CPU in container), Pillow 12.1.0 + pillow_heif 1.7.0, waitress 3.0.2, Docker (`python:3.10-slim`), GitHub Actions, vanilla JS frontend.

**Spec:** `docs/superpowers/specs/2026-09-16-phase-a-deployability-design.md`

## Global Constraints

- Interpreter: `D:\CV+DLPROJECT\venv\Scripts\python.exe`. From Bash always `cd "/d/CV+DLPROJECT" && venv/Scripts/python.exe ...` — never the global Python (it has no torch/flask).
- Test command: `cd "/d/CV+DLPROJECT" && venv/Scripts/python.exe -m pytest tests/ -q -p no:cacheprovider`. **Baseline before Task 1: `29 passed`.** Read the count every run, not just the exit code.
- Docker is NOT available on the dev machine. Container claims are verified only by the Task 9 GitHub Actions run.
- Per-file upload limit **10 MB**; request cap **90 MB**; frontend chunk **8 files**; `MAX_BATCH_IMAGES` stays **50**.
- Allowed formats: `JPEG, PNG, WEBP, BMP, AVIF, HEIF`.
- New pins: `pillow_heif==1.7.0`, `waitress==3.0.2` (resolved by `pip install --dry-run` on 2026-09-16).
- Release tag `model-v1`, asset `phase2_ema.pth`, repo `Pranav-1201/Explanable-Hybrid-Computer-Vision-System-for-Robust-Scene-Understanding` (public).
- **Two outward-facing gates require Pranav's explicit yes in chat:** publishing the release asset (Task 4) and pushing to trigger CI (Task 9). Stop and ask; do not batch them with other steps.
- No retraining. Served predictions must be numerically identical to today's EMA model (Task 2 parity check).
- Commits: one per task, message written to a scratchpad file with the editor and committed with `git commit -F <file>`. **No AI attribution lines.** After each commit run `git log -1 --format=%s | head -c 8 | xxd` (no `ef bb bf`) and `git status --short` (delete any 0-byte junk files after reading them).
- Env vars introduced: `MODEL_PATH`, `MODEL_URL`, `STRICT_STARTUP`, `CORS_ORIGINS`, `HOST`, `PORT`, `THREADS`.
- `<scratchpad>` in commands means the executing session's scratchpad directory (never the repo).
- **Deviations from the spec, decided while planning:** (1) the spec's "classes.json equals checkpoint output dimension" and "11 MB file inside a batch while siblings succeed" checks cannot be hermetic unit tests (the checkpoint is gitignored and `app` loads it at import), so they are enforced at startup (`ArtifactError` on mismatch) and verified live in Task 2 Step 5 / Task 5 Step 7; (2) the ResNet-18 / `phase2_best.pth` serving fallback is removed (Task 4 Step 6) so a missing artifact can never silently degrade to a different model.

## File Map

| File | Status | Responsibility |
|---|---|---|
| `scripts/verify2.py`, `scripts/debug_hybrid.py`, `scripts/classesPrint.py`, `scripts/pipeline_timer.py` | moved | scratch/research scripts, run from repo root |
| `serving/__init__.py` | create | package marker |
| `serving/artifacts.py` | create | `ArtifactError`, `load_classes`, `sha256_file`, `load_manifest`, `ensure_weights` |
| `serving/uploads.py` | create | upload limits, `UploadError`, `load_validated_image`, HEIF opener registration |
| `serving/config.py` | create | `env_flag`, `env_int`, `cors_origins` |
| `scripts/export_serving_artifacts.py` | create | one-off: slim EMA checkpoint + classes.json + manifest into `dist/`, with parity check |
| `models/classes.json` | create | 67 sorted class names (committed) |
| `models/serving_manifest.json` | create | weights filename, release URL, sha256, size (committed) |
| `models/cnn_baseline.py` | modify | `pretrained` flag: build skeleton without reading/downloading weights |
| `app.py` | modify | use `serving.*`; lock; strict startup; CORS from env |
| `frontend/index.html` | modify | chunked batch upload |
| `serve.py` | create | waitress entry point |
| `requirements.txt` | modify | add pillow_heif, waitress |
| `requirements-serve.txt`, `requirements-test.txt` | create | container dependency sets |
| `Dockerfile`, `.dockerignore` | create | `serve` and `test` targets |
| `scripts/smoke_test.py` | create | container smoke client (stdlib + Pillow) |
| `.github/workflows/docker-smoke.yml` | create | CI acceptance job |
| `scripts/field_rerun.py` | create | re-run the E:\ field folders through a live server |
| `README.md` | modify | "Run from a fresh clone" section |
| `tests/test_serving_artifacts.py`, `tests/test_cnn_baseline_skeleton.py`, `tests/test_uploads.py`, `tests/test_serving_config.py` | create | hermetic tests (no model, no dataset) |

`app.py` calls `load_models()` at import and needs the checkpoint, so **no test may import `app`**. Everything testable lives in `serving/`.

---

### Task 1: Move scratch scripts (B11)

**Files:**
- Move: `verify2.py`, `debug_hybrid.py`, `classesPrint.py`, `pipeline_timer.py` into `scripts/`
- Delete: `results.txt`
- Modify: `run_all.bat:15`

**Interfaces:** none.

- [ ] **Step 1: Confirm the only references**

Run: `cd "/d/CV+DLPROJECT" && git grep -n "verify2\|debug_hybrid\|classesPrint\|pipeline_timer\|results.txt" -- ':!FAANG_AUDIT_ROADMAP.md' ':!docs/'`
Expected: only `run_all.bat:15:python pipeline_timer.py`. If anything else appears, update it in Step 3 too.

- [ ] **Step 2: Move and delete with git**

```bash
cd "/d/CV+DLPROJECT" && mkdir -p scripts && git mv verify2.py scripts/verify2.py && git mv debug_hybrid.py scripts/debug_hybrid.py && git mv classesPrint.py scripts/classesPrint.py && git mv pipeline_timer.py scripts/pipeline_timer.py && git rm -q results.txt
```

- [ ] **Step 3: Update `run_all.bat` line 15**

Replace `python pipeline_timer.py` with `python scripts\pipeline_timer.py` (the batch file already does `cd /d %~dp0`, so paths inside the script still resolve from the repo root).

- [ ] **Step 4: Verify**

Run: `cd "/d/CV+DLPROJECT" && git status --short && venv/Scripts/python.exe -m pytest tests/ -q -p no:cacheprovider 2>&1 | tail -2`
Expected: four `R` renames, one `D results.txt`, `M run_all.bat`; `29 passed`.

- [ ] **Step 5: Commit** — message file content:

```
chore(repo): move scratch scripts to scripts/, drop stale results.txt (B11)

verify2.py, debug_hybrid.py, classesPrint.py and pipeline_timer.py were
research scratch files in the repo root. run_all.bat updated to the new
pipeline_timer path. results.txt was a stale 19.63% baseline dump that no
longer describes any served model; git history keeps it.
```

---

### Task 2: Committed class list and slim serving checkpoint (B1, part of B2)

**Files:**
- Create: `serving/__init__.py`, `serving/artifacts.py`, `scripts/export_serving_artifacts.py`, `models/classes.json` (generated), `tests/test_serving_artifacts.py`
- Modify: `app.py:89-90` (`DATA_DIR`), `app.py:166-172` (`discover_classes`), `app.py:178`, `app.py:359-361`
- Modify: `.gitignore` (add `dist/`)

**Interfaces:**
- Produces: `serving.artifacts.ArtifactError(RuntimeError)`; `load_classes(path: str) -> list[str]`; `sha256_file(path: str) -> str`.
- Produces: `dist/phase2_ema.pth` = `{"state_dict": <EMA state dict>, "backbone": str, "val_acc": float, "best_is_ema": True, "num_classes": int}`; `dist/manifest.json` = `{"weights": {"filename": "phase2_ema.pth", "sha256": str, "size": int}, "num_classes": int}`.

- [ ] **Step 1: Write the failing tests** — `tests/test_serving_artifacts.py`:

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd "/d/CV+DLPROJECT" && venv/Scripts/python.exe -m pytest tests/test_serving_artifacts.py -q -p no:cacheprovider 2>&1 | tail -3`
Expected: collection error `ModuleNotFoundError: No module named 'serving'`.

- [ ] **Step 3: Implement `serving/__init__.py` and `serving/artifacts.py`**

`serving/__init__.py`:

```python
"""Serving-side helpers that do not depend on Flask routes or the dataset."""
```

`serving/artifacts.py`:

```python
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
```

- [ ] **Step 4: Write the export script** — `scripts/export_serving_artifacts.py`:

```python
"""Export the serving artifacts from the Phase-2 training checkpoint.

Writes into dist/:
  phase2_ema.pth  - EMA weights only (no optimizer, scheduler or raw weights)
  classes.json    - sorted MIT Indoor class names, as training indexed them
  manifest.json   - sha256 and byte size of the weights

Then proves the slim checkpoint serves identical logits to the full one.
Run from the repo root: venv/Scripts/python.exe scripts/export_serving_artifacts.py
"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # CNNBaseline reads models/... relative to the working directory

import torch  # noqa: E402

from models.cnn_baseline import CNNBaseline  # noqa: E402
from serving.artifacts import sha256_file  # noqa: E402
from utils.checkpoint import load_checkpoint  # noqa: E402

SRC = os.path.join(ROOT, "models", "phase2_best.pth")
TRAIN_DIR = os.path.join(ROOT, "data", "MIT_Indoor", "train")
OUT = os.path.join(ROOT, "dist")


def main() -> None:
    ckpt = torch.load(SRC, map_location="cpu", weights_only=True)
    if not ckpt.get("best_is_ema"):
        raise SystemExit("phase2_best.pth has best_is_ema=False; serving expects EMA weights")
    state = ckpt["ema_state"]
    head = [k for k in state if k.endswith("fc.5.weight")]
    if len(head) != 1:
        raise SystemExit(f"expected exactly one classifier weight, found {head}")
    num_classes = int(state[head[0]].shape[0])

    classes = sorted(d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d)))
    if len(classes) != num_classes:
        raise SystemExit(f"{len(classes)} dataset classes but head has {num_classes} outputs")

    os.makedirs(OUT, exist_ok=True)
    weights_path = os.path.join(OUT, "phase2_ema.pth")
    torch.save({"state_dict": state, "backbone": ckpt["backbone"],
                "val_acc": float(ckpt["val_acc"]), "best_is_ema": True,
                "num_classes": num_classes}, weights_path)
    with open(os.path.join(OUT, "classes.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(classes, f, indent=2)
        f.write("\n")
    manifest = {"weights": {"filename": "phase2_ema.pth", "sha256": sha256_file(weights_path),
                            "size": os.path.getsize(weights_path)},
                "num_classes": num_classes}
    with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    full = CNNBaseline(num_classes, backbone=ckpt["backbone"])
    load_checkpoint(SRC, full, device="cpu", load_ema=True)
    slim = CNNBaseline(num_classes, backbone=ckpt["backbone"])
    load_checkpoint(weights_path, slim, device="cpu")
    torch.manual_seed(0)
    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        diff = (full(x) - slim(x)).abs().max().item()

    print(f"source   {SRC}  {os.path.getsize(SRC)} bytes")
    print(f"weights  {weights_path}  {manifest['weights']['size']} bytes")
    print(f"sha256   {manifest['weights']['sha256']}")
    print(f"classes  {num_classes}")
    print(f"parity   max|full-slim| logits = {diff}")
    if diff != 0.0:
        raise SystemExit("PARITY FAILED: slim checkpoint does not reproduce the served model")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the export and read every line**

Run: `cd "/d/CV+DLPROJECT" && venv/Scripts/python.exe scripts/export_serving_artifacts.py`
Expected: `classes  67`, `parity   max|full-slim| logits = 0.0`, and a weights size **measured here** (record the exact bytes in the commit message; do not estimate). If `fc.5.weight` is not found, print `list(state)[-6:]` and fix the suffix — do not guess.

- [ ] **Step 6: Commit the class list and switch `app.py` to it**

Run: `cd "/d/CV+DLPROJECT" && cp dist/classes.json models/classes.json`. Then, **with the editor** (not a shell redirect), append to `.gitignore`:

```
# Serving artifacts built by scripts/export_serving_artifacts.py
dist/
```

In `app.py`, replace `DATA_DIR   = os.path.join(ROOT, "data", "MIT_Indoor")` with:

```python
CLASSES_PATH = os.path.join(MODELS_DIR, "classes.json")
```

Delete `discover_classes()` (lines 166-172). Add near the other imports:

```python
from serving.artifacts import ArtifactError, load_classes
```

In `load_models()` replace `classes     = discover_classes()` with:

```python
    try:
        classes = load_classes(CLASSES_PATH)
    except ArtifactError as e:
        print(f"[WARN] {e}")
        classes = []
```

In `predict()` replace the 503 message `"No classes found. Check data/MIT_Indoor/train exists."` with `"No classes loaded. Check models/classes.json."`.

- [ ] **Step 7: Verify tests, class-order identity, and a live prediction**

Run: `cd "/d/CV+DLPROJECT" && venv/Scripts/python.exe -m pytest tests/ -q -p no:cacheprovider 2>&1 | tail -2`
Expected: `38 passed` (29 existing + 9 new: 1 + 6 parametrized + 1 + 1). Report the number as printed.

Run: `cd "/d/CV+DLPROJECT" && venv/Scripts/python.exe -c "import json,os; d='data/MIT_Indoor/train'; print(json.load(open('models/classes.json'))==sorted(x for x in os.listdir(d) if os.path.isdir(os.path.join(d,x))))"`
Expected: `True`.

Start `venv/Scripts/python.exe app.py` in the background, then `curl -s -F image=@data/MIT_Indoor/test/bedroom/$(ls data/MIT_Indoor/test/bedroom | head -1) http://127.0.0.1:5000/predict | venv/Scripts/python.exe -c "import sys,json; r=json.load(sys.stdin); print(r.get('prediction'), r.get('confidence'))"`. Expected: a bedroom-family prediction with a confidence; `/health` shows `num_classes: 67`. Stop the server.

- [ ] **Step 8: Commit** — message file:

```
feat(serve): load classes from committed models/classes.json; add artifact export (B1)

The server no longer needs data/MIT_Indoor/train to know its class names.
models/classes.json is the sorted 67-class list training indexed by, verified
identical to the old directory scan. serving/artifacts.load_classes rejects
empty, unsorted or duplicated lists, which would silently mislabel outputs.

scripts/export_serving_artifacts.py writes an EMA-only checkpoint
(<SIZE> bytes vs 394201078 for phase2_best.pth) plus a sha256 manifest into
dist/, and checks logits parity against the full checkpoint (max diff 0.0).
```

Replace `<SIZE>` with the measured byte count from Step 5 before committing. Stage explicitly: `git add serving/__init__.py serving/artifacts.py scripts/export_serving_artifacts.py models/classes.json tests/test_serving_artifacts.py app.py .gitignore`.

---

### Task 3: Build the served backbone without pretrained weights

**Files:**
- Modify: `models/cnn_baseline.py:10-14`
- Test: `tests/test_cnn_baseline_skeleton.py`

**Interfaces:**
- Produces: `CNNBaseline(num_classes: int = 67, backbone: str = 'resnet50_places365_local', pretrained: bool = True)`. With `pretrained=False` it reads no file and makes no network call; the caller must load a full checkpoint afterwards.

- [ ] **Step 1: Write the failing test** — `tests/test_cnn_baseline_skeleton.py`:

```python
"""Serving builds the architecture only; weights come from the checkpoint.

Before this, constructing the served backbone read the gitignored Places365
.tar via a CWD-relative path and fell back to downloading ImageNet weights -
both immediately overwritten by the checkpoint, both fatal in a container.
"""
import pytest
import torch
import torchvision.models._api as tv_api

from models.cnn_baseline import CNNBaseline


def _forbid(*_args, **_kwargs):
    raise AssertionError("pretrained weights must not be read or downloaded")


@pytest.fixture
def no_weights(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # models/resnet50_places365_weights.pth.tar unreachable
    monkeypatch.setattr(tv_api, "load_state_dict_from_url", _forbid)
    monkeypatch.setattr(torch.hub, "load", _forbid)


def test_skeleton_builds_without_weights(no_weights):
    model = CNNBaseline(67, backbone="resnet50_places365_local", pretrained=False)
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 224, 224))
    assert out.shape == (1, 67)


def test_default_still_loads_pretrained(no_weights):
    with pytest.raises(AssertionError, match="must not be read"):
        CNNBaseline(67, backbone="resnet50_places365_local")


def test_skeleton_rejects_unknown_backbone():
    with pytest.raises(ValueError):
        CNNBaseline(67, backbone="vgg16", pretrained=False)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd "/d/CV+DLPROJECT" && venv/Scripts/python.exe -m pytest tests/test_cnn_baseline_skeleton.py -q -p no:cacheprovider 2>&1 | tail -4`
Expected: `test_skeleton_builds_without_weights` FAILS with `TypeError: ... unexpected keyword argument 'pretrained'`; `test_default_still_loads_pretrained` PASSES (proves the forbid hook actually intercepts the download — if it does not pass, the patch target is wrong; stop and find the real call site before continuing).

- [ ] **Step 3: Implement** — in `models/cnn_baseline.py` change the signature and the head of the if-chain:

```python
    def __init__(self, num_classes: int = 67, backbone: str = 'resnet50_places365_local',
                 pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone

        if not pretrained:
            # Architecture only: the caller loads a full checkpoint on top, so
            # reading or downloading pretrained weights here is wasted work and
            # a hard dependency on files/network a container does not have.
            if backbone.startswith('resnet50'):
                self.model = models.resnet50(weights=None)
            elif backbone == 'resnet18_imagenet':
                self.model = models.resnet18(weights=None)
            else:
                raise ValueError(f"Unknown backbone: {backbone}")
        elif backbone == 'resnet50_places365':
```

(The existing `if backbone == 'resnet50_places365':` becomes the `elif` above; the rest of the chain is unchanged.)

- [ ] **Step 4: Run tests**

Run: `cd "/d/CV+DLPROJECT" && venv/Scripts/python.exe -m pytest tests/ -q -p no:cacheprovider 2>&1 | tail -2`
Expected: previous count + 3, all passed.

- [ ] **Step 5: Commit** — message file:

```
feat(model): CNNBaseline(pretrained=False) builds the architecture only

Serving constructs the ResNet-50 and then loads a full checkpoint over it, so
the pretrained weights were always discarded. They were still read from the
gitignored models/resnet50_places365_weights.pth.tar via a CWD-relative path,
with a silent fallback to downloading ImageNet weights - a hidden file and
network dependency for any container. Training call sites keep the default.
```

---

### Task 4: Checksummed weight download and release publish (B2)

**Files:**
- Modify: `serving/artifacts.py` (add `load_manifest`, `ensure_weights`)
- Create: `models/serving_manifest.json`
- Modify: `app.py:61-87` (`build_baseline_model`), `app.py:175-191` (`load_models`)
- Test: `tests/test_serving_artifacts.py` (append)

**Interfaces:**
- Consumes: `CNNBaseline(..., pretrained=False)` (Task 3); `dist/manifest.json`, `dist/phase2_ema.pth` (Task 2).
- Produces: `load_manifest(path: str) -> dict` returning `{"filename": str, "url": str, "sha256": str, "size": int}`; `ensure_weights(path: str, url: str, sha256: str, size: int, fetch=urllib.request.urlopen) -> str`.

- [ ] **Step 1: Append failing tests** to `tests/test_serving_artifacts.py` (merge the new imports into the import block at the top of the file):

```python
import hashlib
import io

from serving.artifacts import ensure_weights, load_manifest

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
```

- [ ] **Step 2: Run to verify failure**

Run: `cd "/d/CV+DLPROJECT" && venv/Scripts/python.exe -m pytest tests/test_serving_artifacts.py -q -p no:cacheprovider 2>&1 | tail -3`
Expected: `ImportError: cannot import name 'ensure_weights'`.

- [ ] **Step 3: Implement** — append to `serving/artifacts.py` (add `import os`, `import shutil`, `import tempfile`, `import urllib.request` to the imports):

```python
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
```

- [ ] **Step 4: Create `models/serving_manifest.json`** using the sha256 and size printed by Task 2 Step 5 / `dist/manifest.json` (copy them from that file, do not retype from memory):

```json
{
  "weights": {
    "filename": "phase2_ema.pth",
    "url": "https://github.com/Pranav-1201/Explanable-Hybrid-Computer-Vision-System-for-Robust-Scene-Understanding/releases/download/model-v1/phase2_ema.pth",
    "sha256": "<copy dist/manifest.json weights.sha256>",
    "size": 0
  }
}
```

Set `size` to `dist/manifest.json` `weights.size`. Verify the copy mechanically: `cd "/d/CV+DLPROJECT" && venv/Scripts/python.exe -c "import json; a=json.load(open('dist/manifest.json'))['weights']; b=json.load(open('models/serving_manifest.json'))['weights']; print(a['sha256']==b['sha256'], a['size']==b['size'])"` must print `True True`.

- [ ] **Step 5: Run tests** — `pytest tests/ -q` as in Global Constraints. Expected: all pass, count = previous + 6.

- [ ] **Step 6: Wire `app.py` to the slim artifact**

Replace `build_baseline_model` (lines 61-87) with:

```python
def build_baseline_model(num_classes: int):
    """Return (loaded_model, gradcam_target_layer) for the served checkpoint.

    Weights are the EMA-only export described in models/serving_manifest.json.
    They are downloaded and sha256-verified on first start when absent, so a
    fresh clone or container needs neither the dataset nor hand-placed files.
    MODEL_PATH relocates the file (e.g. a container volume); MODEL_URL
    overrides the download source. The checksum is enforced either way.
    """
    from models.cnn_baseline import CNNBaseline
    from utils.checkpoint import load_checkpoint

    manifest = load_manifest(MANIFEST_PATH)
    path = os.environ.get("MODEL_PATH") or os.path.join(MODELS_DIR, manifest["filename"])
    url = os.environ.get("MODEL_URL") or manifest["url"]
    ensure_weights(path, url, manifest["sha256"], manifest["size"])

    meta = torch.load(path, map_location="cpu", weights_only=True)
    if int(meta["num_classes"]) != num_classes:
        raise ArtifactError(f"{path} has {meta['num_classes']} outputs but "
                            f"classes.json lists {num_classes} classes")
    model = CNNBaseline(num_classes, backbone=meta["backbone"], pretrained=False)
    load_checkpoint(path, model, device=DEVICE)
    print(f"[INFO] Serving {meta['backbone']} EMA from {path} "
          f"(val_acc={meta['val_acc']:.2f}%)")
    return model, model.model.layer4[-1]
```

Move `MODELS_DIR` above this function and add `MANIFEST_PATH = os.path.join(MODELS_DIR, "serving_manifest.json")`. Extend the import to `from serving.artifacts import ArtifactError, ensure_weights, load_classes, load_manifest`.

Replace the model block in `load_models()` (the `phase2_path`/`baseline_path` existence check and its try/except) with:

```python
    try:
        if not classes:
            raise ArtifactError("no classes loaded; cannot size the model head")
        baseline_model, baseline_target_layer = build_baseline_model(len(classes))
        baseline_model.to(DEVICE).eval()
    except Exception as e:
        print(f"[WARN] Could not load baseline model: {e}")
        baseline_model = None
        baseline_target_layer = None
```

Delete the now-unused `num_classes = max(len(classes), 1)`. The ResNet-18 `baseline.pth` / `phase2_best.pth` fallback is intentionally removed: the served artifact is always obtainable from the release, and a silent fallback to a 70.82% model is the failure this phase exists to prevent.

- [ ] **Step 7: Place the local copy and verify serving is unchanged**

Run: `cd "/d/CV+DLPROJECT" && cp dist/phase2_ema.pth models/phase2_ema.pth` then start `venv/Scripts/python.exe app.py` in the background and repeat the Task 2 Step 7 `/predict` call on the same bedroom image. Expected: log line `Serving resnet50_places365_local EMA from ...phase2_ema.pth`; same prediction **and same confidence** as Task 2 Step 7. Stop the server.

- [ ] **Step 8: GATE — publish the release asset (ask Pranav first)**

Stop and ask in chat: "Ready to publish `dist/phase2_ema.pth` (<SIZE> bytes, sha256 <first 12 chars>) publicly as release `model-v1`. Have you checked the Places365 / MIT Indoor licence terms? Publish?" Proceed only on an explicit yes. Then write release notes to a scratchpad file and run:

```bash
cd "/d/CV+DLPROJECT" && gh release create model-v1 dist/phase2_ema.pth --title "Serving weights model-v1" --notes-file "<scratchpad>/release_notes.md"
```

Notes content: backbone, EMA, test top-1 83.21 (EMA) / 83.88 (EMA+TTA, served), T=0.5142, sha256, size, "exported by scripts/export_serving_artifacts.py from phase2_best.pth epoch 25".

- [ ] **Step 9: Verify the published asset end-to-end**

Run: `cd "/d/CV+DLPROJECT" && venv/Scripts/python.exe -c "from serving.artifacts import load_manifest, ensure_weights; m=load_manifest('models/serving_manifest.json'); import tempfile,os; d=tempfile.mkdtemp(); p=os.path.join(d,'w.pth'); ensure_weights(p, m['url'], m['sha256'], m['size']); print('downloaded+verified', os.path.getsize(p))"`
Expected: `downloaded+verified <SIZE>`. Delete the temp dir afterwards.

- [ ] **Step 10: Commit** — message file:

```
feat(serve): fetch serving weights from release model-v1 with sha256 check (B2)

models/serving_manifest.json pins the EMA-only checkpoint's URL, size and
sha256. serving.artifacts.ensure_weights downloads it on first start into a
same-directory temp file and renames only after verification; a corrupt
existing file is reported rather than silently replaced. app.py builds the
backbone with pretrained=False and loads the verified artifact, checking the
head size against classes.json. The phase2_best.pth / ResNet-18 fallback is
removed so a missing artifact can never degrade to a different model.
```

---

### Task 5: Per-file upload limit and chunked batch upload (B3)

**Files:**
- Create: `serving/uploads.py`, `tests/test_uploads.py`
- Modify: `app.py:95-122` (limits, `UploadError`), `app.py:240-298` (`load_validated_image`, 413 handler)
- Modify: `frontend/index.html:469-484` (`handleFiles`)

**Interfaces:**
- Produces: `serving.uploads` constants `MAX_UPLOAD_BYTES = 10 MiB`, `FILES_PER_REQUEST = 8`, `MAX_REQUEST_BYTES = 90 MiB`, `ALLOWED_FORMATS`, `MIN_SIDE_PX`, `MAX_SIDE_PX`, `MAX_TOTAL_PIXELS`; `class UploadError(ValueError)` with `.message`, `.status`; `load_validated_image(file_storage) -> PIL.Image.Image`.
- Produces (frontend): `const CHUNK_SIZE = 8;` — must equal `FILES_PER_REQUEST`.

- [ ] **Step 1: Write the failing tests** — `tests/test_uploads.py`:

```python
"""Upload validation, importable without the model (app.py loads it at import)."""
import io
import os
import re

import pytest
from PIL import Image
from werkzeug.datastructures import FileStorage

from serving import uploads
from serving.uploads import UploadError, load_validated_image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _image(fmt="JPEG", size=(64, 48)):
    buf = io.BytesIO()
    Image.new("RGB", size, (120, 80, 40)).save(buf, format=fmt)
    return buf.getvalue()


def _fs(data, name="x.jpg"):
    return FileStorage(stream=io.BytesIO(data), filename=name)


def test_valid_jpeg_is_returned():
    assert load_validated_image(_fs(_image())).size == (64, 48)


def test_empty_file_is_400():
    with pytest.raises(UploadError) as e:
        load_validated_image(_fs(b""))
    assert e.value.status == 400


def test_file_over_per_file_limit_is_413():
    with pytest.raises(UploadError) as e:
        load_validated_image(_fs(b"\0" * (uploads.MAX_UPLOAD_BYTES + 1)))
    assert e.value.status == 413


def test_non_image_is_400():
    with pytest.raises(UploadError) as e:
        load_validated_image(_fs(b"not an image at all"))
    assert e.value.status == 400


def test_too_small_is_400():
    with pytest.raises(UploadError):
        load_validated_image(_fs(_image(size=(16, 16))))


def test_request_cap_fits_a_full_chunk_of_max_size_files():
    assert uploads.MAX_REQUEST_BYTES >= uploads.FILES_PER_REQUEST * uploads.MAX_UPLOAD_BYTES


def test_frontend_chunk_size_matches_server():
    with open(os.path.join(ROOT, "frontend", "index.html"), encoding="utf-8") as f:
        m = re.search(r"const CHUNK_SIZE = (\d+);", f.read())
    assert m, "frontend must declare const CHUNK_SIZE"
    assert int(m.group(1)) == uploads.FILES_PER_REQUEST
```

- [ ] **Step 2: Run to verify failure**

Run: `cd "/d/CV+DLPROJECT" && venv/Scripts/python.exe -m pytest tests/test_uploads.py -q -p no:cacheprovider 2>&1 | tail -3`
Expected: `ModuleNotFoundError: No module named 'serving.uploads'`.

- [ ] **Step 3: Create `serving/uploads.py`** by moving the existing code verbatim from `app.py` (limits block lines 95-108, `UploadError` 111-121, `load_validated_image` 240-291), then apply these changes:

```python
"""Upload validation for /predict and /predict_batch (B-12, B3).

Every upload is decoded to prove it is an image of an allowed format; the
filename and content-type are never trusted. Limits are per file; the Flask
request cap (MAX_REQUEST_BYTES) only has to admit one frontend chunk.
"""
import os

from PIL import Image

MAX_UPLOAD_BYTES  = 10 * 1024 * 1024        # per file
FILES_PER_REQUEST = 8                       # frontend CHUNK_SIZE must match
MAX_REQUEST_BYTES = FILES_PER_REQUEST * MAX_UPLOAD_BYTES + 10 * 1024 * 1024  # + multipart headroom
ALLOWED_FORMATS   = {"JPEG", "PNG", "WEBP", "BMP"}
MIN_SIDE_PX       = 32
MAX_SIDE_PX       = 10_000
MAX_TOTAL_PIXELS  = 40_000_000

# Pillow raises DecompressionBombError past this instead of allocating the pixels.
Image.MAX_IMAGE_PIXELS = MAX_TOTAL_PIXELS
```

In `load_validated_image`, change the oversize message to `f"Image exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB per-file limit."` (status stays 413). Everything else is moved unchanged.

- [ ] **Step 4: Rewire `app.py`**

Delete the moved blocks from `app.py`. Add:

```python
from serving.uploads import (MAX_REQUEST_BYTES, UploadError, load_validated_image)
```

Replace `app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES` with `app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_BYTES` and the 413 handler body with:

```python
    return jsonify({"error": f"Request exceeds the {MAX_REQUEST_BYTES // (1024 * 1024)} MB "
                             f"limit; send fewer photos per request."}), 413
```

- [ ] **Step 5: Chunk the frontend batch** — in `frontend/index.html` replace `handleFiles` (lines 469-484) with:

```js
  // Must equal FILES_PER_REQUEST in serving/uploads.py (tests/test_uploads.py guards it).
  const CHUNK_SIZE = 8;

  async function postChunk(chunk) {
    const form = new FormData();
    chunk.forEach((f) => form.append("images", f));
    const failed = (msg) => ({
      results: chunk.map((f) => ({ filename: f.name, error: msg, review: true, review_reason: "error" })),
    });
    try {
      const r = await fetch("/predict_batch", { method: "POST", body: form });
      const data = await r.json().catch(() => null);
      if (r.ok && data && data.results) return data;
      return failed((data && data.error) || `Server returned ${r.status}.`);
    } catch (_) {
      return failed("Could not reach the tagger. Is the server running?");
    }
  }

  async function handleFiles(list) {
    const picked = Array.from(list).slice(0, 50);
    if (!picked.length) return;
    picked.forEach((f) => files.push(f));
    $("empty").classList.add("hidden");
    $("results").classList.remove("hidden");
    const grid = $("grid");
    grid.setAttribute("aria-busy", "true");
    grid.innerHTML = `<div class="dz-note" style="grid-column:1/-1">Tagging ${picked.length} photo(s)…</div>`;

    // Sequential chunks keep each request under the server's body cap; results
    // are appended in upload order so render()'s file indexing stays aligned.
    const merged = [];
    let threshold = null;
    for (let i = 0; i < picked.length; i += CHUNK_SIZE) {
      const data = await postChunk(picked.slice(i, i + CHUNK_SIZE));
      merged.push(...data.results);
      if (data.threshold != null) threshold = data.threshold;
    }
    const tagged = merged.filter((it) => it.in_scope).length;
    render({ results: merged, threshold,
             summary: { n: merged.length, tagged, review: merged.length - tagged } });
  }
```

- [ ] **Step 6: Run tests**

Run the full suite. Expected: all pass, count = previous + 7.

- [ ] **Step 7: Live verification (server + browser)**

Start `venv/Scripts/python.exe app.py` in the background. Write this scratch script with the editor to the session scratchpad as `check_limits.py` (not committed) and run it with the venv interpreter:

```python
"""Scratch: per-file limit becomes an inline row; request cap still 413s."""
import io
import json
import os
import urllib.error
import urllib.request
import uuid

from PIL import Image

BASE = "http://127.0.0.1:5000"


def jpeg(seed):
    buf = io.BytesIO()
    Image.new("RGB", (320, 240), (seed * 40 % 256, 90, 60)).save(buf, format="JPEG")
    return buf.getvalue()


def send(files):
    boundary = uuid.uuid4().hex
    body = b"".join(
        (f'--{boundary}\r\nContent-Disposition: form-data; name="images"; filename="{n}"\r\n'
         f"Content-Type: application/octet-stream\r\n\r\n").encode() + d + b"\r\n"
        for n, d in files) + f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(BASE + "/predict_batch", data=body, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


big = b"\xff\xd8" + os.urandom(11 * 1024 * 1024)
status, data = send([("a.jpg", jpeg(1)), ("b.jpg", jpeg(2)), ("c.jpg", jpeg(3)), ("big.jpg", big)])
print("mixed:", status, [(r["filename"], r.get("review_reason"), r.get("error")) for r in data["results"]])

status, data = send([(f"h{i}.jpg", b"\xff\xd8" + os.urandom(int(9.5 * 1024 * 1024))) for i in range(10)])
print("oversize request:", status, data)
```

Expected output:
1. `mixed: 200` with 4 rows: `big.jpg` has `review_reason` `invalid` and the "per-file limit" message; `a/b/c.jpg` have no error.
2. `oversize request: 413` with the "send fewer photos per request" message.

In the browser (in-app browser or claude-in-chrome), load `http://127.0.0.1:5000/`, then run in the page:

```js
(async () => {
  const input = document.getElementById("file-input");
  const dt = new DataTransfer();
  for (let i = 0; i < 20; i++) {
    const c = Object.assign(document.createElement("canvas"), { width: 200, height: 150 });
    const g = c.getContext("2d"); g.fillStyle = `hsl(${i * 18},60%,50%)`; g.fillRect(0, 0, 200, 150);
    const blob = await new Promise((res) => c.toBlob(res, "image/jpeg"));
    dt.items.add(new File([blob], `p${i}.jpg`, { type: "image/jpeg" }));
  }
  input.files = dt.files;
  input.dispatchEvent(new Event("change", { bubbles: true }));
})();
```

Expected (network panel / request log): **3** POSTs to `/predict_batch` (8 + 8 + 4 files), all 200; summary total reads **20**; 20 cards. Stop the server.

- [ ] **Step 8: Commit** — message file:

```
fix(upload): enforce 10 MB per file and chunk batches of 8 (B3)

MAX_CONTENT_LENGTH bounded the whole request at 10 MB, so the promised
50-photo batch 413'd on real photos. Validation moves to serving/uploads.py
(importable without loading the model) and the size limit is now per file;
the request cap is 8 x 10 MB + 10 MB headroom. The frontend sends batches as
sequential requests of 8, merges results in upload order, and marks only a
failed chunk's cards as errors instead of blanking the grid. A test pins the
frontend CHUNK_SIZE to the server's FILES_PER_REQUEST.
```

---

### Task 6: Accept AVIF and HEIC (M1, F3)

**Files:**
- Modify: `serving/uploads.py`, `requirements.txt`, `tests/test_uploads.py`

**Interfaces:**
- Produces: `ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "AVIF", "HEIF"}`; HEIF opener registered on import of `serving.uploads`.

- [ ] **Step 1: Append failing tests** to `tests/test_uploads.py`:

```python
@pytest.mark.parametrize("fmt", ["AVIF", "HEIF"])
def test_phone_formats_decode(fmt):
    img = load_validated_image(_fs(_image(fmt), name=f"x.{fmt.lower()}"))
    assert img.convert("RGB").size == (64, 48)


def test_unsupported_format_message_is_actionable():
    with pytest.raises(UploadError) as e:
        load_validated_image(_fs(_image("GIF"), name="x.gif"))
    assert "HEIF" in e.value.message and "convert" in e.value.message.lower()
```

Note: `_image("HEIF")` needs the HEIF opener, which `serving.uploads` registers on import in Step 4.

- [ ] **Step 2: Install the dependency and run to verify failure**

Run: `cd "/d/CV+DLPROJECT" && venv/Scripts/python.exe -m pip install pillow_heif==1.7.0 2>&1 | tail -1 && venv/Scripts/python.exe -m pytest tests/test_uploads.py -q -p no:cacheprovider 2>&1 | tail -5`
Expected: `test_phone_formats_decode[AVIF]` FAILS with `Unsupported image format 'AVIF'`; `[HEIF]` FAILS (`KeyError: 'HEIF'` on save, because no opener is registered yet); the message test FAILS.

- [ ] **Step 3: Record the dependency** — add `pillow_heif==1.7.0` to `requirements.txt` in alphabetical position (after `pillow==12.1.0`).

- [ ] **Step 4: Implement** in `serving/uploads.py`:

```python
import pillow_heif

# iPhone default capture format. AVIF is decoded natively by Pillow 12.
pillow_heif.register_heif_opener()

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "AVIF", "HEIF"}
```

and the unsupported-format error:

```python
    if fmt not in ALLOWED_FORMATS:
        raise UploadError(
            f"Unsupported image format {fmt or 'unknown'!r}. "
            f"Accepted: {', '.join(sorted(ALLOWED_FORMATS))}. "
            f"Convert the photo to JPEG and try again.")
```

- [ ] **Step 5: Run the full suite.** Expected: all pass, count = previous + 3.

- [ ] **Step 6: Commit** — message file:

```
feat(upload): accept AVIF and HEIC phone photos (M1, F3)

AVIF was rejected by our own allowlist - Pillow 12.1 already decodes it.
HEIC needs pillow_heif (1.7.0, registered on import of serving.uploads).
The unsupported-format message now lists accepted formats and says to
convert. Tests generate AVIF/HEIF fixtures in memory; nothing is committed.
```

---

### Task 7: waitress, inference lock, strict startup, CORS from env (B4)

**Files:**
- Create: `serving/config.py`, `serve.py`, `tests/test_serving_config.py`
- Modify: `app.py` (imports, `CORS(app)`, `load_models`, `/predict`, `/predict_batch`, module bottom), `requirements.txt`

**Interfaces:**
- Produces: `env_flag(name: str, environ=os.environ) -> bool`; `env_int(name: str, default: int, environ=os.environ) -> int`; `cors_origins(environ=os.environ)` returning `"*"` or `list[str]`; `app.INFERENCE_LOCK: threading.Lock`; `load_models(strict: bool) -> None`.

- [ ] **Step 1: Write failing tests** — `tests/test_serving_config.py`:

```python
import pytest

from serving.config import cors_origins, env_flag, env_int


@pytest.mark.parametrize("raw,expected", [("1", True), ("true", True), ("ON", True),
                                          ("", False), ("0", False), ("no", False)])
def test_env_flag(raw, expected):
    assert env_flag("X", {"X": raw}) is expected


def test_env_flag_unset_is_false():
    assert env_flag("X", {}) is False


def test_env_flag_rejects_garbage():
    with pytest.raises(ValueError):
        env_flag("X", {"X": "maybe"})


def test_env_int():
    assert env_int("P", 5000, {}) == 5000
    assert env_int("P", 5000, {"P": "7860"}) == 7860
    with pytest.raises(ValueError):
        env_int("P", 5000, {"P": "abc"})


def test_cors_origins():
    assert cors_origins({}) == "*"
    assert cors_origins({"CORS_ORIGINS": "*"}) == "*"
    assert cors_origins({"CORS_ORIGINS": "https://a.example, https://b.example"}) == [
        "https://a.example", "https://b.example"]
```

- [ ] **Step 2: Run to verify failure** — `ModuleNotFoundError: No module named 'serving.config'`.

- [ ] **Step 3: Implement `serving/config.py`**

```python
"""Environment configuration for serving. Invalid values fail loudly."""
import os

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"", "0", "false", "no", "off"}


def env_flag(name: str, environ=os.environ) -> bool:
    raw = environ.get(name, "").strip().lower()
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    raise ValueError(f"{name}={raw!r} is not a boolean (use 1/0, true/false, yes/no, on/off)")


def env_int(name: str, default: int, environ=os.environ) -> int:
    raw = environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not an integer") from None


def cors_origins(environ=os.environ):
    """'*' (today's behaviour) unless CORS_ORIGINS lists comma-separated origins."""
    raw = environ.get("CORS_ORIGINS", "*").strip()
    if raw in ("", "*"):
        return "*"
    return [o.strip() for o in raw.split(",") if o.strip()]
```

- [ ] **Step 4: Run tests.** Expected: previous + 10 passed (6 parametrized + 4).

- [ ] **Step 5: Wire `app.py`**

Imports: add `import threading` and `from serving.config import cors_origins, env_flag`. Replace `CORS(app)` with `CORS(app, origins=cors_origins())`.

After the global model handles add:

```python
# One model, shared by every waitress thread. Grad-CAM registers forward and
# backward hooks on it, and concurrent forward passes would interleave those
# hooks, so all model execution is serialised. Upload parsing and validation
# still run concurrently. On CPU this costs little throughput.
INFERENCE_LOCK = threading.Lock()
```

Change `def load_models():` to `def load_models(strict: bool = False):` and the model try/except to:

```python
    except Exception as e:
        if strict:
            raise
        print(f"[WARN] Could not load baseline model: {e}")
        baseline_model = None
        baseline_target_layer = None
```

Also in strict mode a failed `load_classes` must raise; that block becomes:

```python
    try:
        classes = load_classes(CLASSES_PATH)
    except ArtifactError as e:
        if strict:
            raise
        print(f"[WARN] {e}")
        classes = []
```

Replace the bare `load_models()` call with:

```python
# STRICT_STARTUP=1 (set in the container) turns a missing or unverifiable
# artifact into a startup failure instead of a server that only returns 503s.
load_models(strict=env_flag("STRICT_STARTUP"))
```

In `/predict`, wrap from `if tta_param != "0":` through the Grad-CAM `try/except` in `with INFERENCE_LOCK:` (indent that block one level). Change the 503 text `"Baseline model not loaded. Run: python training/train_baseline.py"` to `"Model not loaded; see the server log."`.

In `/predict_batch`, wrap only the `if use_tta: ... else: ...` probability computation in `with INFERENCE_LOCK:` so batches from different clients interleave per image.

- [ ] **Step 6: Create `serve.py`**

```python
"""Production entry point: the Flask app under waitress (Windows and Linux).

    venv/Scripts/python.exe serve.py            # dev machine
    python serve.py                             # container (PORT/THREADS from env)

`python app.py` remains the Flask development server.
"""
import os

from waitress import serve

from serving.config import env_int


def main() -> None:
    from app import app  # loads the model; STRICT_STARTUP makes failures fatal

    host = os.environ.get("HOST", "0.0.0.0")
    port = env_int("PORT", 5000)
    threads = env_int("THREADS", 4)
    print(f"[INFO] waitress serving on http://{host}:{port} with {threads} threads")
    serve(app, host=host, port=port, threads=threads)


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Install and record waitress**

Run: `cd "/d/CV+DLPROJECT" && venv/Scripts/python.exe -m pip install waitress==3.0.2 2>&1 | tail -1`. Add `waitress==3.0.2` to `requirements.txt` in alphabetical position.

- [ ] **Step 8: Live verification — three checks, each with printed output**

(a) Normal start: `venv/Scripts/python.exe serve.py` in the background; `curl -s http://127.0.0.1:5000/health` shows `baseline_loaded: true`, `num_classes: 67`.

(b) Concurrency: a scratch script sends 6 simultaneous `/predict` requests (ThreadPoolExecutor, 6 workers, same real image) and prints each status and whether `gradcam` is present / `gradcam_error` absent. Expected: 6 × `200 gradcam=True error=None`, identical predictions. Stop the server.

(c) Strict failure: 
```bash
cd "/d/CV+DLPROJECT" && STRICT_STARTUP=1 MODEL_PATH="<scratchpad>/missing/w.pth" MODEL_URL="http://127.0.0.1:9/none" venv/Scripts/python.exe serve.py; echo "exit=$?"
```
Expected: traceback ending in `ArtifactError: download failed from http://127.0.0.1:9/none`, `exit=1`, and no leftover `.part` file under `<scratchpad>/missing/`. Then the same without `STRICT_STARTUP` starts and `/health` shows `baseline_loaded: false` (lenient path preserved); stop it.

- [ ] **Step 9: Run the full suite**, then commit — message file:

```
feat(serve): waitress entry point, inference lock, strict startup (B4)

serve.py runs the app under waitress on Windows and in the container.
All model execution (TTA forwards and Grad-CAM, whose hooks live on the
shared model) is serialised by INFERENCE_LOCK; verified with 6 concurrent
/predict calls returning Grad-CAM and identical predictions.
STRICT_STARTUP=1 makes a missing or unverifiable artifact a startup failure
instead of a 503-only server. CORS origins come from CORS_ORIGINS, defaulting
to the previous allow-all (tightening is B7). serving/config.py parses env
values and rejects malformed ones.
```

---

### Task 8: Container definition (B5)

**Files:**
- Create: `requirements-serve.txt`, `requirements-test.txt`, `Dockerfile`, `.dockerignore`

**Interfaces:**
- Produces: Docker targets `serve` (default, runs `python serve.py` on `PORT=7860`) and `test` (runs pytest). Port 7860 is the Hugging Face Spaces default for Phase F.

Docker cannot run on this machine; this task's container behaviour is verified by Task 9. Say so in the commit message.

- [ ] **Step 1: `requirements-serve.txt`** — pins are the venv's resolved closure of the serving imports (computed 2026-09-16 from `importlib.metadata`), torch excluded because the Dockerfile installs CPU wheels first:

```
# Serving dependencies for the CPU container. torch==2.5.1 and
# torchvision==0.20.1 are installed first from the PyTorch CPU index (Dockerfile).
blinker==1.9.0
click==8.4.2
contourpy==1.3.2
cycler==0.12.1
flask==3.1.3
flask-cors==6.0.2
fonttools==4.61.1
grad-cam==1.5.5
itsdangerous==2.2.0
jinja2==3.1.6
joblib==1.5.3
kiwisolver==1.4.9
markupsafe==3.0.3
matplotlib==3.10.8
numpy==2.2.6
opencv-python==4.13.0.90
packaging==26.0
pillow==12.1.0
pillow_heif==1.7.0
pyparsing==3.3.2
python-dateutil==2.9.0.post0
scikit-learn==1.7.2
scipy==1.15.3
six==1.17.0
threadpoolctl==3.6.0
tqdm==4.67.3
ttach==0.0.3
waitress==3.0.2
werkzeug==3.1.7
```

- [ ] **Step 2: `requirements-test.txt`**

```
# Test-only additions for the container's `test` target.
-r requirements-serve.txt
exceptiongroup==1.3.1
imageio==2.37.2
iniconfig==2.3.0
lazy-loader==0.4
networkx==3.4.2
pluggy==1.6.0
pygments==2.20.0
pytest==9.1.1
scikit-image==0.25.2
tifffile==2025.5.10
tomli==2.4.1
typing-extensions==4.15.0
```

- [ ] **Step 3: `.dockerignore`**

```
.git/
.github/
.claude/
.claude-flow/
venv/
.env/
**/__pycache__/
**/*.pyc
dist/
docs/
data/MIT_Indoor/
data/MITINDOOR/
data/Raw/
data/features/
data/train/
data/test/
**/*.npz
**/*.pth
**/*.pt
**/*.pth.tar
**/*.pkl
outputs/
logs/
results/
```

- [ ] **Step 4: `Dockerfile`**

```dockerfile
# CPU serving image for the interior tagger.
#   docker build -t cvdl .                 # serve target (default)
#   docker build --target test -t cvdl-test .
FROM python:3.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# opencv-python (a grad-cam dependency) needs libGL and glib at import time.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 torchvision==0.20.1
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY . .
RUN useradd --create-home --uid 10001 app \
 && mkdir -p /models \
 && chown app:app /models

ENV MODEL_PATH=/models/phase2_ema.pth \
    STRICT_STARTUP=1 \
    HOST=0.0.0.0 \
    PORT=7860 \
    THREADS=4

FROM base AS test
COPY requirements-test.txt .
RUN pip install -r requirements-test.txt
USER app
CMD ["python", "-m", "pytest", "tests/", "-q", "-p", "no:cacheprovider"]

FROM base AS serve
USER app
VOLUME ["/models"]
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ['PORT'], timeout=4)" || exit 1
CMD ["python", "serve.py"]
```

- [ ] **Step 5: Static checks available locally**

Run: `cd "/d/CV+DLPROJECT" && git check-ignore -v models/phase2_ema.pth dist/phase2_ema.pth` (both ignored by git) and `venv/Scripts/python.exe -c "import re; t=open('requirements-serve.txt').read(); print(len([l for l in t.splitlines() if re.match(r'^[A-Za-z]', l)]), 'pins')"` — expected `29 pins`. Confirm every name in `requirements-serve.txt` appears in `requirements.txt` (case/underscore-insensitive) with the same version via a scratch script; print mismatches (expected: none).

- [ ] **Step 6: Commit** — message file:

```
build(docker): CPU serve and test images (B5)

python:3.10-slim with CPU torch 2.5.1 / torchvision 0.20.1, serving deps
pinned to the venv's resolved versions, non-root user, /models volume for
the downloaded weights, STRICT_STARTUP=1, HEALTHCHECK on /health, port 7860.
A test target adds pytest and scikit-image and runs the suite in-image.
.dockerignore keeps the dataset, venv and all checkpoints out of the build
context, so the image cannot accidentally ship local weights.

Not built on the dev machine (no Docker); verified by the docker-smoke CI job.
```

---

### Task 9: GitHub Actions smoke job (container acceptance)

**Files:**
- Create: `scripts/smoke_test.py`, `.github/workflows/docker-smoke.yml`

**Interfaces:**
- Consumes: Docker targets from Task 8; `/health`, `/predict`, `/predict_batch`; `FILES_PER_REQUEST = 8`.
- Produces: `scripts/smoke_test.py` helpers `multipart(files, fields=None) -> tuple[bytes, str]` and `post(url, body, content_type, timeout) -> tuple[int, dict]` (reused by Task 10).

- [ ] **Step 1: `scripts/smoke_test.py`**

```python
"""Container smoke test: health, single predictions for JPEG/AVIF/HEIF, and a
20-photo batch sent in chunks of 8. Stdlib HTTP plus Pillow for fixtures.

    python scripts/smoke_test.py --base http://127.0.0.1:7860
"""
import argparse
import io
import json
import sys
import time
import urllib.error
import urllib.request
import uuid

import pillow_heif
from PIL import Image

pillow_heif.register_heif_opener()
CHUNK = 8


def image_bytes(fmt: str, seed: int) -> bytes:
    colour = ((seed * 53) % 256, (seed * 97) % 256, (seed * 29) % 256)
    buf = io.BytesIO()
    Image.new("RGB", (320, 240), colour).save(buf, format=fmt)
    return buf.getvalue()


def multipart(files, fields=None):
    """files: iterable of (field, filename, bytes). Returns (body, content_type)."""
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in (fields or {}).items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
                     f"{value}\r\n".encode())
    for field, filename, data in files:
        head = (f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; '
                f'filename="{filename}"\r\nContent-Type: application/octet-stream\r\n\r\n')
        parts.append(head.encode() + data + b"\r\n")
    body = b"".join(parts) + f"--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def post(url, body, content_type, timeout):
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": content_type})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def wait_healthy(base, timeout_s):
    deadline, last = time.monotonic() + timeout_s, None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(base + "/health", timeout=5) as r:
                health = json.load(r)
            if health.get("baseline_loaded"):
                return health
            last = health
        except (urllib.error.URLError, OSError) as e:
            last = e
        time.sleep(3)
    raise SystemExit(f"FAIL: not healthy after {timeout_s}s (last: {last})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:7860")
    ap.add_argument("--health-timeout", type=int, default=600)
    ap.add_argument("--request-timeout", type=int, default=300)
    args = ap.parse_args()

    started = time.monotonic()
    health = wait_healthy(args.base, args.health_timeout)
    classes = set(health["classes"])
    print(f"health ok after {time.monotonic() - started:.0f}s: {len(classes)} classes")
    failures = []

    for fmt in ("JPEG", "AVIF", "HEIF"):
        body, ctype = multipart([("image", f"probe.{fmt.lower()}", image_bytes(fmt, 1))])
        t0 = time.monotonic()
        status, data = post(args.base + "/predict", body, ctype, args.request_timeout)
        label = data.get("original_prediction") or data.get("prediction")
        ok = status == 200 and label in classes
        print(f"/predict {fmt}: status={status} label={label} {time.monotonic() - t0:.1f}s {'OK' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"/predict {fmt}: {status} {data}")

    rows = []
    for start in range(0, 20, CHUNK):
        chunk = [("images", f"b{i}.jpg", image_bytes("JPEG", i)) for i in range(start, min(start + CHUNK, 20))]
        body, ctype = multipart(chunk)
        status, data = post(args.base + "/predict_batch", body, ctype, args.request_timeout)
        print(f"/predict_batch chunk@{start}: status={status} rows={len(data.get('results', []))}")
        if status != 200:
            failures.append(f"/predict_batch chunk@{start}: {status} {data}")
        rows.extend(data.get("results", []))
    errored = [r for r in rows if r.get("error")]
    if len(rows) != 20 or errored:
        failures.append(f"batch: {len(rows)} rows, {len(errored)} errored")

    print(f"SUMMARY: {len(failures)} failure(s)")
    for f in failures:
        print("  -", f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run it locally against waitress** — start `venv/Scripts/python.exe serve.py` (port 5000) in the background, then `cd "/d/CV+DLPROJECT" && venv/Scripts/python.exe scripts/smoke_test.py --base http://127.0.0.1:5000`. Expected: 3 `/predict` OK lines, 3 batch chunks (8, 8, 4 rows), `SUMMARY: 0 failure(s)`, exit 0. Then prove it can fail: stop the server, run it with `--health-timeout 10`, expect `FAIL: not healthy` and a non-zero exit.

- [ ] **Step 3: `.github/workflows/docker-smoke.yml`**

```yaml
name: docker-smoke

on:
  push:
    branches: [main]
  pull_request:
  workflow_dispatch:

jobs:
  smoke:
    runs-on: ubuntu-latest
    timeout-minutes: 60
    steps:
      - uses: actions/checkout@v4

      - name: Build serve image
        run: docker build --target serve -t cvdl-serve .

      - name: Build test image
        run: docker build --target test -t cvdl-test .

      - name: Image carries no dataset and no weights
        run: |
          docker run --rm --entrypoint sh cvdl-serve -c '
            test ! -e data/MIT_Indoor &&
            test ! -e /models/phase2_ema.pth &&
            test -z "$(find /app /models \( -name "*.pth" -o -name "*.pth.tar" -o -name "*.pkl" \) 2>/dev/null)" &&
            echo "clean: no dataset, no checkpoints"'

      - name: Unit tests inside the image
        run: docker run --rm cvdl-test

      - name: Start container
        run: docker run -d --name cvdl -p 7860:7860 cvdl-serve

      - name: Smoke test against the running container
        run: docker run --rm --network host cvdl-test python scripts/smoke_test.py --base http://127.0.0.1:7860 --health-timeout 600

      - name: Container logs
        if: always()
        run: docker logs cvdl || true
```

Confirm the trigger fires on the branch being pushed: this repo pushes straight to `main`, which is in `branches`.

- [ ] **Step 4: Commit locally** — message file:

```
ci: docker-smoke job builds and exercises the container (Phase A acceptance)

Builds serve and test targets, asserts the image holds no dataset and no
checkpoint, runs the unit suite in-image, starts the container so it must
download and verify weights from release model-v1, then runs
scripts/smoke_test.py: /health, /predict for JPEG, AVIF and HEIF, and a
20-photo batch in chunks of 8. The smoke script passed locally against
waitress and was seen failing with no server running.
```

- [ ] **Step 5: GATE — push (ask Pranav first)**

Ask: "Tasks 1–9 are committed locally (N commits). Push to `origin/main` to trigger docker-smoke?" On yes: `git push origin main`.

- [ ] **Step 6: Confirm the run exists, then watch it**

Run: `cd "/d/CV+DLPROJECT" && gh run list --workflow docker-smoke.yml --limit 1`. An empty list is NOT "pending" — re-check the workflow's branch filter and use `gh workflow run docker-smoke.yml --ref main` if needed. Then `gh run watch <id> --exit-status`.
Expected: all steps green. Read the log: `clean: no dataset, no checkpoints`; pytest count equal to the local count; `health ok after Ns`; `SUMMARY: 0 failure(s)`. If a step fails, use superpowers:systematic-debugging; each fix is its own commit and needs another push (ask again).

---

### Task 10: Field re-run and README

**Files:**
- Create: `scripts/field_rerun.py`
- Modify: `README.md` (new section before `## ⚙️ Installation`)

**Interfaces:**
- Consumes: `multipart`, `post` from `scripts/smoke_test.py`.

- [ ] **Step 1: `scripts/field_rerun.py`**

```python
"""Re-run real field-photo folders through a live server and write a CSV per
folder in the frontend's export format. Reports rows rejected for format.

    python scripts/field_rerun.py --base http://127.0.0.1:5000 --out <dir> <folder> [<folder> ...]
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smoke_test import CHUNK, multipart, post  # noqa: E402

HEADER = ["filename", "room_tag", "prediction", "confidence", "in_scope", "review_reason", "error"]


def run_folder(base, folder, out_dir):
    names = sorted(n for n in os.listdir(folder) if os.path.isfile(os.path.join(folder, n)))
    rows = []
    for start in range(0, len(names), CHUNK):
        files = []
        for n in names[start:start + CHUNK]:
            with open(os.path.join(folder, n), "rb") as f:
                files.append(("images", n, f.read()))
        body, ctype = multipart(files)
        status, data = post(base + "/predict_batch", body, ctype, 600)
        if status != 200:
            raise SystemExit(f"{folder} chunk@{start}: HTTP {status} {data}")
        rows.extend(data["results"])
    out = os.path.join(out_dir, os.path.basename(os.path.normpath(folder)).replace(" ", "_") + ".csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for r in rows:
            w.writerow([r.get("filename"), r.get("label", ""), r.get("prediction", ""),
                        r.get("confidence", ""), r.get("in_scope", False),
                        r.get("review_reason") or "", r.get("error", "")])
    format_rejects = [r for r in rows if "Unsupported image format" in (r.get("error") or "")]
    invalid = [r for r in rows if r.get("review_reason") == "invalid"]
    tagged = sum(1 for r in rows if r.get("in_scope"))
    print(f"{folder}: files={len(names)} tagged={tagged} review={len(rows) - tagged} "
          f"invalid={len(invalid)} format_rejects={len(format_rejects)} -> {out}")
    for r in invalid:
        print(f"    invalid: {r['filename']}: {r.get('error')}")
    return len(format_rejects)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:5000")
    ap.add_argument("--out", required=True)
    ap.add_argument("folders", nargs="+")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    total = sum(run_folder(args.base, f, args.out) for f in args.folders)
    print(f"TOTAL format rejects: {total}")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run the acceptance check** — with `venv/Scripts/python.exe serve.py` running:

```bash
cd "/d/CV+DLPROJECT" && P="/e/Projects and Research papers/CV + DL Project - Explainable Hybrid Computer Vision System for Robust Scene Understanding" && venv/Scripts/python.exe scripts/field_rerun.py --base http://127.0.0.1:5000 --out "<scratchpad>/field" "$P/testing" "$P/Testing dataset"
```

Expected: `testing: files=10`, `Testing dataset: files=62` (72 files: 44 jpg, 19 jpeg, 4 avif, 4 webp, 1 png as listed on 2026-09-16), `TOTAL format rejects: 0`, exit 0. Every `invalid:` line printed must be read and reported with its reason — a non-format invalid (corrupt/too small) is reported, not hidden. Show the per-folder summary lines and the four AVIF rows from the CSVs.

- [ ] **Step 3: README section** — insert before `## ⚙️ Installation`:

````markdown
## 🚀 Run from a fresh clone

The server needs **neither the dataset nor hand-placed weights**. On first start
it downloads the EMA-only checkpoint from release
[`model-v1`](https://github.com/Pranav-1201/Explanable-Hybrid-Computer-Vision-System-for-Robust-Scene-Understanding/releases/tag/model-v1) and verifies its sha256 against
`models/serving_manifest.json`; class names come from `models/classes.json`.

**Docker (CPU):**

```bash
docker build -t cvdl .
docker run -p 7860:7860 -v cvdl-models:/models cvdl
# open http://localhost:7860
```

**Local venv:**

```bash
pip install -r requirements.txt
python serve.py            # waitress on http://localhost:5000
```

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_PATH` | `models/phase2_ema.pth` (`/models/phase2_ema.pth` in Docker) | where the weights live / are downloaded to |
| `MODEL_URL` | release `model-v1` asset | alternative download source (checksum still enforced) |
| `STRICT_STARTUP` | off (`1` in Docker) | exit on missing/unverifiable artifacts instead of serving 503s |
| `CORS_ORIGINS` | `*` | comma-separated allowed origins |
| `HOST` / `PORT` / `THREADS` | `0.0.0.0` / `5000` (`7860` in Docker) / `4` | waitress binding |

Accepted uploads: JPEG, PNG, WEBP, BMP, AVIF, HEIC — up to 10 MB each, 50 per batch.
The container is built, unit-tested and smoke-tested on every push by
`.github/workflows/docker-smoke.yml`.
````

Also add a badge line under the title: `![docker-smoke](https://github.com/Pranav-1201/Explanable-Hybrid-Computer-Vision-System-for-Robust-Scene-Understanding/actions/workflows/docker-smoke.yml/badge.svg)`.

- [ ] **Step 4: Commit** — message file:

```
docs(readme): run from a fresh clone; add field re-run script (Phase A)

README documents Docker and venv startup, the artifact download, and every
serving environment variable. scripts/field_rerun.py pushes real photo
folders through a live server in chunks and writes frontend-format CSVs.
Acceptance on the two E:\ field folders (72 files incl. 4 AVIF):
format rejects 0.
```

Update the commit text with the actual numbers printed in Step 2 before committing. Push is covered by the Task 9 gate; if Tasks 9-10 are pushed separately, ask again.

---

## Phase A exit checklist (all with fresh evidence in the final report)

- [ ] Full local suite count and `passed` line.
- [ ] Task 2 parity `0.0` and slim checkpoint byte size.
- [ ] Release `model-v1` exists and verifies by download (Task 4 Step 9).
- [ ] 20-photo browser batch = 3 requests, 20 cards (Task 5 Step 7).
- [ ] Strict-startup `exit=1` and lenient `baseline_loaded: false` (Task 7 Step 8c).
- [ ] docker-smoke run URL, green, with `clean: no dataset, no checkpoints` and `SUMMARY: 0 failure(s)` in its log.
- [ ] Field re-run `TOTAL format rejects: 0`.
- [ ] Memory `hybrid-cv-session-status.md` updated with Phase A outcome.
