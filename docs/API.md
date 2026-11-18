# API reference

Base URL: `http://localhost:5000` (waitress, `serve.py`) or `http://localhost:7860` (Docker). All responses are JSON.

## `GET /health`

Liveness + model status. Used by the Docker `HEALTHCHECK` and `scripts/smoke_test.py`.

```json
{
  "status": "ok",
  "device": "cuda",
  "baseline_loaded": true,
  "num_classes": 67,
  "classes": ["airport_inside", "artstudio", "..."]
}
```

`baseline_loaded: false` means the model failed to load (lenient startup, `STRICT_STARTUP=0`) — every prediction endpoint will 503 until it's fixed.

## `GET /classes`

```json
{
  "classes": ["airport_inside", "artstudio", "..."],
  "home_labels": {"kitchen": "Kitchen", "bedroom": "Bedroom", "...": "..."}
}
```

`home_labels` maps each of the 24 in-scope class ids to its display name. It is the single source of truth for the frontend's correction dropdown (never copy the list client-side).

The full 67-class MIT Indoor label set the model was trained on, in class-index order (index 0 = `airport_inside`, etc.). Only 24 of these are in-scope for the product (`HOME_CLASS_LABELS`, `app.py`); everything else routes to review as `out_of_scope`.

## `POST /predict`

Single image, multipart form field **`image`**. Returns the prediction plus a Grad-CAM explanation.

| Field | Type | Default | Notes |
|---|---|---|---|
| `image` | file | required | JPEG, PNG, WEBP, BMP, AVIF or HEIC/HEIF, ≤ 10 MB |
| `model` | form field | `baseline` | only `baseline` is served; fusion is a disabled research arm |
| `tta` | query/form | `1` | `0` disables test-time augmentation (7 forward passes → 1, faster, ~0.67pt less accurate) |

Response (trimmed — `gradcam` is a base64 PNG, shortened here; the upload is not echoed back):

```json
{
  "model": "baseline",
  "prediction": "kitchen",
  "confidence": 0.91,
  "calibrated": true,
  "temperature": 0.5142,
  "tta_used": true,
  "top5": [{"class": "kitchen", "prob": 0.91}, "..."],
  "out_of_scope": false,
  "gradcam": "<base64 PNG>"
}
```

If test-time augmentation or Grad-CAM fails, the request still returns `200` without that part, and `tta_error` or `gradcam_error` holds a generic message ending in `error_id=<12 hex>`; the traceback is in the server log under the same id. The exception text itself is never returned.

If `confidence` falls below the rejection threshold, `prediction` is overwritten to `"Unknown / Out of Scope"`, `out_of_scope` becomes `true`, and the real call is preserved in `original_prediction`. The threshold is loaded from `models/calibration_config.json`'s `rejection_threshold` at startup (**0.478**, temperature-calibrated — see README); `0.30` is only the fallback used if that file is missing.

Errors: `400` no file / unsupported format / bad request; `413` file or request too large; `429` rate limit exceeded (see below); `503` model or classes not loaded; `500` unhandled (returns only an `error_id` — never a traceback — logged server-side).

**Rate limit.** `/predict` and `/predict_batch` allow `30 per minute` per client address by default (`PREDICT_RATE_LIMIT`). `/health`, `/classes` and `/metrics` are not limited. The limiter keys on the socket address, so behind a reverse proxy every visitor shares one bucket (audit X13, not yet fixed). A `429` currently comes back as an HTML page, not JSON (audit X11, not yet fixed).

## `POST /predict_batch`

Multiple images, multipart field **`images`** (repeated). Built for the frontend's review-queue workflow; **no** Grad-CAM per image (too slow at batch scale — call `/predict` on a single card if you need one).

- Hard cap: 50 files per request (`MAX_BATCH_IMAGES`). In practice, size requests to **≤ 8 files** — the server's request-body cap (`MAX_REQUEST_BYTES`) is only sized for `FILES_PER_REQUEST=8` files at the 10 MB/file limit plus multipart overhead; the frontend and `scripts/smoke_test.py`/`field_rerun.py` chunk accordingly.
- Per-file failures (bad format, too small, corrupt) don't abort the batch — they come back as a row with an `error` field.

```json
{
  "results": [
    {
      "filename": "kitchen1.jpg",
      "prediction": "kitchen",
      "label": "Kitchen",
      "confidence": 0.91,
      "in_scope": true,
      "review": false,
      "review_reason": null,
      "top5": ["..."]
    },
    {
      "filename": "corrupt.jpg",
      "error": "File is not a readable image (it may be truncated or corrupt).",
      "review": true,
      "review_reason": "invalid"
    }
  ],
  "summary": {"n": 2, "tagged": 1, "review": 1},
  "threshold": 0.478,
  "tta_used": true
}
```

`review_reason` is one of `null` (in scope), `"low_confidence"` (below threshold), `"out_of_scope"` (predicted class not in the 24 home classes), or `"invalid"` (failed upload validation).

## `GET /metrics`

In-process latency and review-rate counters since the process started (lost on restart). Public and unauthenticated. The numbers below are illustrative.

```json
{
  "predict":       {"requests": 12, "p50_ms": 140.2, "p95_ms": 610.0},
  "predict_batch": {"requests": 3,  "p50_ms": 900.5, "p95_ms": 1210.0},
  "tagged": 20,
  "reviewed": 8,
  "review_rate": 0.2857
}
```

A latency series appears only after its endpoint has been called once; `review_rate` is `null` until a photo has been classified.

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `MODEL_PATH` | `models/phase2_ema.pth` (`/models/phase2_ema.pth` in Docker) | where the weights live or are downloaded to |
| `MODEL_URL` | release `model-v1` asset | alternative download source (checksum still enforced) |
| `STRICT_STARTUP` | off (`1` in Docker) | exit on missing or unverifiable artifacts instead of serving 503s |
| `CORS_ORIGINS` | `*` | comma-separated allowed origins |
| `HOST` / `PORT` / `THREADS` | `0.0.0.0` / `5000` (`7860` in Docker) / `4` | waitress binding |
| `PREDICT_RATE_LIMIT` | `30 per minute` | limit on `/predict` and `/predict_batch` |
| `REVIEW_MARGIN_MIN` | unset (rule off) | send a photo to review when top-1 minus top-2 probability is below this |
| `REVIEW_ENTROPY_MAX` | unset (rule off) | send a photo to review when normalised prediction entropy (0 = one-hot, 1 = uniform) exceeds this |

Both `REVIEW_*` values must lie in 0 to 1; a malformed value stops startup rather than being ignored.
## `GET /`

Serves `frontend/index.html` (the drag-drop UI), same origin as the API.

## Errors

Every error response is `{"error": "<message>"}`, optionally with `"error_id"` for 500s. No stack traces are ever returned to the client (audit finding N14).
