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
{ "classes": ["airport_inside", "artstudio", "..."] }
```

The full 67-class MIT Indoor label set the model was trained on, in class-index order (index 0 = `airport_inside`, etc.). Only 24 of these are in-scope for the product (`HOME_CLASS_LABELS`, `app.py`); everything else routes to review as `out_of_scope`.

## `POST /predict`

Single image, multipart form field **`image`**. Returns the prediction plus a Grad-CAM explanation.

| Field | Type | Default | Notes |
|---|---|---|---|
| `image` | file | required | JPEG, PNG, WEBP, BMP, AVIF or HEIC/HEIF, ≤ 10 MB |
| `model` | form field | `baseline` | only `baseline` is served; fusion is a disabled research arm |
| `tta` | query/form | `1` | `0` disables test-time augmentation (7 forward passes → 1, faster, ~0.67pt less accurate) |

Response (trimmed — `gradcam` and `original_image` are base64 PNGs, omitted here):

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
  "gradcam": "<base64 PNG>",
  "original_image": "<base64 PNG>"
}
```

If `confidence` falls below the rejection threshold, `prediction` is overwritten to `"Unknown / Out of Scope"`, `out_of_scope` becomes `true`, and the real call is preserved in `original_prediction`. The threshold is loaded from `models/calibration_config.json`'s `rejection_threshold` at startup (**0.478**, temperature-calibrated — see README); `0.30` is only the fallback used if that file is missing.

Errors: `400` no file / unsupported format / bad request; `503` model or classes not loaded; `500` unhandled (returns only an `error_id` — never a traceback — logged server-side).

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

## `GET /`

Serves `frontend/index.html` (the drag-drop UI), same origin as the API.

## Errors

Every error response is `{"error": "<message>"}`, optionally with `"error_id"` for 500s. No stack traces are ever returned to the client (audit finding N14).
