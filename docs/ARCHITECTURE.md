# Architecture

The shape of the system, not its implementation. For request-by-request detail
see [FLOW.md](FLOW.md); for the HTTP contract see [API.md](API.md).

Verified against the tree on 2026-09-29 (commit `dff2c9b` plus the uncommitted
Phase B work listed in [../HANDOVER.md](../HANDOVER.md)).

## What this is

A scene classifier (MIT Indoor 67) scoped into a real-estate **interior tagger**:
a photo is either auto-tagged with a room name or sent to a human review queue.
The research code (hybrid CNN + handcrafted features) and the served product
live in one repo; only the CNN path is served.

## Two halves

```
RESEARCH (offline, not served)              SERVING (what the container runs)
------------------------------              ---------------------------------
data/            dataset loaders, splits    app.py            Flask app, endpoints
preprocessing/   HOG / CNN feature extract  serve.py          waitress entry point
classical_features/  HOG, edges, corners..  serving/          config, uploads, routing,
training/        train_phase2 (served),                         scope, metrics, artifacts
                 fusion, svm, scoped (M3)   inference/tta.py  test-time augmentation
calibration/     temperature scaling, ECE   models/           architectures + small JSON
evaluation/      ablation, field eval,                          artifacts (weights excluded)
                 robustness                 frontend/         single-file UI, no build step
explainability/  Grad-CAM (also used live)  utils/            checkpoint loading
experiments/, demo/, scripts/               tests/            pytest suite
```

## Serving modules

| Module | Responsibility | Depends on |
|---|---|---|
| `app.py` | Endpoints `/`, `/health`, `/classes`, `/metrics`, `/predict`, `/predict_batch`; security headers; rate limiting; the model lock | everything below |
| `serve.py` | Runs `app` under waitress (`PORT`, `THREADS`) | `app` |
| `serving/artifacts.py` | Loads `classes.json`; downloads and sha256-verifies weights named in `models/serving_manifest.json` | stdlib |
| `serving/uploads.py` | Decodes every upload to prove it is an allowed image; size and pixel limits; HEIC/AVIF opener | Pillow, pillow-heif |
| `serving/scope.py` | `HOME_CLASS_LABELS`: which of the 67 classes count as "home" | nothing |
| `serving/routing.py` | Pure function: probabilities + scope -> review reason or `None` | nothing |
| `serving/config.py` | Environment parsing that fails loudly on garbage | stdlib |
| `serving/metrics.py` | In-process p50/p95 latency and tagged-vs-review counts | stdlib |
| `inference/tta.py` | 3-view test-time augmentation (7 forward passes) + temperature scaling | torch |
| `models/cnn_baseline.py` | ResNet-50 with a 512-wide head; `pretrained=False` builds the architecture only | torchvision |

`serving/scope.py` and `serving/routing.py` deliberately import nothing from the
web app, so the tagging decision can be unit-tested and reused by offline
evaluation without loading a model.

## Artifacts and where they live

| Artifact | Location | In git? |
|---|---|---|
| Served weights `phase2_ema.pth` (~99 MB) | GitHub release `model-v1`, fetched at startup, sha256-checked | no |
| `classes.json`, `calibration_config.json`, `temperature_cnn.json`, `serving_manifest.json` | `models/` | yes |
| MIT Indoor dataset | `data/MIT_Indoor/` (local only) | no |
| Field photos (72) | outside the repo, see [CONSTRAINTS.md](CONSTRAINTS.md) | no, by decision |
| Field ground-truth labels | `evaluation/field_labels.csv` | yes |

## Deployment shape

One Docker image, two targets (`serve`, `test`), CPU torch pinned as
`+cpu` builds. The image carries no dataset and no weights; weights arrive on
first start. CI (`ci`, `docker-smoke`) is described in [TEST_CHECKLIST.md](TEST_CHECKLIST.md).

## Known limits of this shape

- The rate limiter and metrics are in-memory: correct for one waitress process,
  per-process (not global) with more. Move both to a shared store before scaling out.
- All model execution is serialised behind `INFERENCE_LOCK` because Grad-CAM
  hooks mutate the shared model.
