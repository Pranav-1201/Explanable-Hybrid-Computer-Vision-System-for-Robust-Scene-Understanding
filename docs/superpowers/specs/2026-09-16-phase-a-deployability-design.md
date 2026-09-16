# Phase A — Deployability & Correctness: Design

**Date:** 2026-09-16 · **Status:** approved in chat, pending written-spec review
**Source plan:** `FAANG_AUDIT_ROADMAP.md`, Phase A (B1, B2, B3, B4, B5, M1, F3, B11)

## Goal and acceptance criteria

A container started on a machine with **no dataset and no pre-placed weights**
serves `/` and returns correct predictions; a 50-photo batch of real phone photos
(including HEIC and AVIF) completes without a 413; pytest is green inside the image.

Measured bar for the two field folders on `E:\` re-run locally through waitress:
**zero rows marked `invalid` because of file format.**

## Findings that shaped this design (verified 2026-09-16, not inherited)

| # | Finding | Evidence |
|---|---|---|
| 1 | AVIF rejection is our allowlist, not Pillow. Pillow 12.1.0 in the venv decodes AVIF. | `features.check('avif')` returned `True`; `ALLOWED_FORMATS` at `app.py:100` omits it |
| 2 | HEIC genuinely needs a plugin. | `import pillow_heif` raised `ModuleNotFoundError` |
| 3 | Third startup blocker, not in the roadmap: building `CNNBaseline('resnet50_places365_local')` reads the gitignored 97 MB `models/resnet50_places365_weights.pth.tar` via a CWD-relative path, and **downloads ImageNet weights from the internet** when it is absent. Both are overwritten by the checkpoint immediately afterwards. | `models/cnn_baseline.py:26-38` |
| 4 | `phase2_best.pth` is 394 MB but serving uses only `ema_state`. | keys: `epoch, model_state, ema_state, optimizer, scheduler, val_acc, val_acc_raw, val_acc_ema, best_is_ema, backbone` |
| 5 | Classes come from the dataset directory (67 dirs). | `discover_classes()` at `app.py:166` |
| 6 | `MAX_CONTENT_LENGTH` (10 MB) bounds the whole request, so a 50-photo batch 413s. | `app.py:106`; frontend posts one `FormData` at `index.html:472-480` |
| 7 | Grad-CAM attaches hooks to the shared model on every request. | `app.py:313` |
| 8 | `load_models()` swallows every load failure; the server starts model-less and 503s. | `app.py:185-191` |
| 9 | No Docker and no WSL distro on the dev machine; `gh` is authenticated; repo is public with no releases. | `docker: command not found`; `gh repo view` visibility PUBLIC |

## Decisions (user-approved 2026-09-16)

- **Weights host:** GitHub Release asset on this repo.
- **Batch limit:** client-side chunking plus a per-file server limit.
- **Formats:** HEIC and AVIF.
- **Container verification:** a GitHub Actions job, since Docker cannot run locally.

## §1 Serving artifacts (B1, B2, finding 3)

- `scripts/export_serving_artifacts.py` (run once, locally) writes into `dist/`:
  - `phase2_ema.pth` — EMA `state_dict` plus `backbone`, `val_acc`, `best_is_ema`; no optimizer, scheduler or raw weights. Its size is measured and recorded, not estimated.
  - `classes.json` — sorted class names, byte-for-byte the list `discover_classes()` returns today.
  - `manifest.json` — sha256 and byte size per artifact.
- `models/classes.json` is committed. The server reads it instead of the dataset and fails when its length differs from the checkpoint's output dimension.
- The slim checkpoint is uploaded as a release asset under tag `model-v1`. **Publishing is outward-facing and requires explicit user confirmation at that step.** Licence terms for redistributing weights derived from Places365 / MIT Indoor are unverified; the user checks them before publishing.
- `serving/artifacts.py`: resolve the weights path; if absent, download from `MODEL_URL` (default: the `model-v1` asset URL), verify sha256 and byte size, write to a temp file and atomically rename. On mismatch, delete the file and raise.
- `CNNBaseline` gains a way to construct the ResNet-50 skeleton with `weights=None` when a full checkpoint will be loaded on top. Training call sites keep their current behaviour.

## §2 Upload path (B3, M1, F3)

- Per-file 10 MB check inside `load_validated_image`, returning a per-file `UploadError` (413 for single `/predict`, an inline `invalid` row inside a batch).
- Request cap `MAX_CONTENT_LENGTH` becomes 8 × 10 MB plus 10 MB multipart headroom = 90 MB.
- Frontend sends the batch as sequential requests of at most 8 files, merges results in original upload order, keeps the single spinner, and on a failed chunk marks only that chunk's cards as errored.
- `ALLOWED_FORMATS` adds `AVIF` and `HEIF`; `pillow_heif.register_heif_opener()` is called once at import.
- The unsupported-format message lists accepted formats and suggests converting to JPEG.

## §3 Runtime (B4)

- `serve.py` runs the app under **waitress** on both Windows and Linux. `python app.py` remains a dev entry point.
- A single module-level `threading.Lock` serialises model execution (TTA forwards and Grad-CAM). Request parsing and validation stay concurrent.
- `STRICT_STARTUP=1` (set in the container): missing weights, checksum mismatch, or class-count mismatch exits non-zero with a clear message. Unset keeps today's lenient local behaviour.
- CORS origins read from `CORS_ORIGINS`, defaulting to today's allow-all. Tightening is B7 (Phase C).

## §4 Container (B5)

- `requirements-serve.txt`: CPU wheels of the same torch/torchvision versions (2.5.1 / 0.20.1) from the PyTorch CPU index; Flask, flask-cors, grad-cam, ttach, pillow, pillow-heif, waitress, numpy, matplotlib, and every transitive import grad-cam needs at runtime (opencv as the headless build), established by importing `app` inside the built image rather than by reading package metadata. Exact pins are taken from the venv, not guessed; any package missing from the venv is resolved and its version recorded.
- `Dockerfile`: `python:3.10-slim` (venv is 3.10.11), non-root user, `HEALTHCHECK` on `/health`, weights downloaded at startup into a `/models` volume so restarts reuse them.
- `.dockerignore` excludes `data/`, `venv/`, `*.pth`, `*.tar`, `*.pkl`, `dist/`, `.claude-flow/`.

## §5 Verification

- `.github/workflows/docker-smoke.yml`, triggered on push to `main` and on pull requests. After the first push, `gh run list` must show the run actually started.
  1. Build the image.
  2. Start it with no dataset and no weights (`STRICT_STARTUP=1`).
  3. Poll `/health` until ready or time out and fail.
  4. POST a JPEG, an AVIF and a HEIC fixture to `/predict`; assert 200 and a class name present in `classes.json`.
  5. POST 20 images through the chunked batch path; assert no 413 and 20 result rows.
  6. Run `pytest` inside the image.
- Fixtures are generated in the job or come from a permissively licensed source recorded next to them; no MIT Indoor images are committed.
- New local tests: `classes.json` equals the checkpoint output dimension; tampered artifact fails the checksum; per-file limit rejects an 11 MB file inside a batch while its siblings succeed; AVIF and HEIF bytes decode through `load_validated_image`.
- Before any fixture test is trusted it is watched failing against the pre-change code.

## §6 Hygiene (B11) and commit sequence

Scratch scripts `verify2.py`, `debug_hybrid.py`, `classesPrint.py`, `pipeline_timer.py` move to `scripts/`; the single caller (`run_all.bat:15`) is updated. `results.txt` (a stale 19.63% baseline dump) is deleted; git history retains it.

One commit per item, each with command-plus-output evidence:

1. B11 move scratch scripts
2. `classes.json` + export script
3. `weights=None` backbone construction for serving
4. artifact download + checksum
5. per-file limit + frontend chunking
6. AVIF / HEIC
7. waitress + inference lock + strict startup
8. Dockerfile + `requirements-serve.txt` + `.dockerignore`
9. CI smoke workflow
10. README "run from a fresh clone" section

## Out of scope

F1 progress UI, B7 rate limiting / headers / CORS tightening, B8 metrics, remaining Phase E CI (ruff, eval gate, LICENSE), Phase F deployment, any retraining.

## Risks

- **Weight licence:** unverified; publishing is gated on user confirmation.
- **pillow-heif wheels:** must install on both Windows (venv) and `python:3.10-slim`; checked at step 6, not assumed.
- **CPU TTA latency in CI:** a GitHub runner is slower than the dev GPU; smoke timeouts are set from the first measured run, not guessed.
