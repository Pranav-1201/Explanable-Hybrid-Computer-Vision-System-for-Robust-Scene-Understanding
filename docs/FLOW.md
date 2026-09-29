# Flow: how a photo becomes a tag

What calls what, in order. Line numbers are omitted on purpose (they rot); each
step names the function or file to open.

## Startup

1. `serve.py:main` imports `app`, which runs `load_models(strict=...)`.
2. `serving/artifacts.py:load_classes` reads `models/classes.json` (67 names).
3. `app.build_baseline_model` -> `load_manifest` -> `ensure_weights`: if the file
   is missing it is downloaded from the release URL and its sha256 and size must
   match `models/serving_manifest.json`, otherwise startup fails.
4. `TemperatureScaler.load` reads `models/temperature_cnn.json` (T = 0.5142).
5. `models/calibration_config.json` supplies `CONFIDENCE_THRESHOLD` (0.478).
6. With `STRICT_STARTUP=1` (the container default) any of the above failing is fatal.

## `POST /predict_batch` (the product path)

```
browser  frontend/index.html
  |  chunks of 8 files (FILES_PER_REQUEST must match the frontend's CHUNK_SIZE)
  v
app.predict_batch                      rate limit: PREDICT_RATE_LIMIT (30/min default)
  |  for each file:
  |-- serving/uploads.py:load_validated_image   decode fully; reject -> per-file
  |                                             {"review_reason": "invalid"}, batch continues
  |-- INFERENCE_LOCK
  |     inference/tta.py:tta_predict            3 views -> temperature-scaled probs, averaged
  |     (or single_predict when ?tta=0)
  |-- serving/routing.py:review_reason          probs + (class in HOME_CLASS_LABELS) + policy
  |       low_confidence -> out_of_scope -> low_margin* -> high_entropy*   (* opt-in)
  |-- serving/metrics.py                        record outcome and latency
  v
JSON: results[], summary, threshold, tta_used
```

`routing_policy()` in `app.py` builds the policy on every call from the live
`CONFIDENCE_THRESHOLD`; `REVIEW_MARGIN_MIN` and `REVIEW_ENTROPY_MAX` switch on
the optional rules.

## `POST /predict` (single image, with explanation)

Same validation and TTA, then Grad-CAM (`run_gradcam`, still under the lock)
and a simpler rule: confidence below the threshold becomes
`"Unknown / Out of Scope"`. It does **not** use `serving/routing.py` and has no
home-class scoping; that is a difference to know about, not an oversight to
"fix" silently.

## Review queue (frontend only)

`frontend/index.html` renders cards from the batch response. A reviewer can
correct a tag from a dropdown fed by `GET /classes` -> `home_labels`, and the CSV
export adds `corrected_tag`. A network or server failure gives a card a
"retry" (`review_reason: "error"`), distinct from a permanent `invalid` file.

## Where a change lands

| If you change... | Touch | Then re-run |
|---|---|---|
| which rooms count as home | `serving/scope.py` only | `tests/test_api_contract.py`, field eval |
| review rules or thresholds | `serving/routing.py`, `models/calibration_config.json` | `tests/test_routing.py`, `evaluation/field_eval.py` |
| the model | `training/`, then `scripts/export_serving_artifacts.py`, `models/serving_manifest.json` | calibration, field eval, docker-smoke |
| upload limits | `serving/uploads.py` (and the frontend chunk size) | `tests/test_uploads.py` |
