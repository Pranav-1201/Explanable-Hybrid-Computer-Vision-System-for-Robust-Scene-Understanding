# Decisions

Why, not just what. One entry per decision that changed the code, newest last.
Every entry names its evidence (a commit or a measurement) and which AI model
and session made it, because behaviour differs between model versions and
"the AI did this" is not a full answer later.

Entries D1 to D6 were reconstructed on 2026-09-29 from commit messages and the
Phase A to D memory notes; the earlier sessions did not record which model
reasoned through them, so that field says "not recorded" rather than guessing.
Do not renumber; add the next `D<n>` at the bottom. Last used id: **D12**.

---

### D1 - Serving weights come from a GitHub release, verified by sha256 (B2)
- **Decision:** `models/serving_manifest.json` pins URL, size and sha256 of the EMA-only checkpoint (release `model-v1`). `ensure_weights` downloads to a temp file and renames only after verification. The old fallback to `phase2_best.pth` / ResNet-18 was removed.
- **Why:** a fresh clone or container has no weights. A silent fallback to a different model would serve wrong answers with no error.
- **Rejected:** committing the 99 MB file (repo bloat); Hugging Face Hub (needs an account: deferred to Phase F).
- **Evidence:** `66c0a29`. The release itself had never been published until 2026-09-24 (it was a 404).
- **By:** not recorded (session 6137fae3, 2026-09-24).

### D2 - CPU torch pinned as `+cpu` builds via `--extra-index-url` (B5)
- **Decision:** `pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.5.1+cpu torchvision==0.20.1+cpu`.
- **Why:** `--index-url` made pip resolve every dependency from the PyTorch index and broke on `typing-extensions`; the fix then left a dependency-confusion gap, closed by the `+cpu` local version, which only the CPU index can satisfy.
- **Evidence:** `c5111bf`, `5182340`; `docker-smoke` green.
- **By:** not recorded (session 6137fae3).

### D3 - Uploads: 10 MB per file, batches of 8, decode fully during validation (B3)
- **Decision:** the request cap is per chunk (8 x 10 MB + headroom), the frontend sends sequential chunks, and `load_validated_image` calls `img.load()` so a truncated photo becomes a per-file 400.
- **Why:** the old whole-request 10 MB cap made the promised 50-photo batch 413. Pillow's `verify()` misses truncated JPEG/HEIF data, so one bad file 500'd a whole batch.
- **Evidence:** `8a505ca`, `480a22d`; a test pins frontend `CHUNK_SIZE` to `FILES_PER_REQUEST`.
- **By:** not recorded.

### D4 - The CSP allows inline script/style and `blob:` images, nothing external
- **Decision:** `default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' blob:`.
- **Why:** the first CSP (`default-src 'self'`) silently blanked the whole frontend: the page is deliberately one file with inline script and style, and thumbnails are `blob:` URLs, which `'self'` does not cover. **No unit test could see this**; it took a real browser run.
- **Consequence:** a frontend change is not verified until it has been loaded in a browser (TEST_CHECKLIST).
- **Evidence:** `dff2c9b`.
- **By:** not recorded.

### D5 - Folder picker instead of recursive drag-and-drop traversal (F4)
- **Decision:** an `<input webkitdirectory>` picker; no directory traversal of dropped folders.
- **Why:** dropped-directory traversal behaves differently across browsers; the picker is consistent.
- **Evidence:** `dff2c9b`.
- **By:** not recorded.

### D6 - All model execution behind one lock
- **Decision:** `INFERENCE_LOCK` serialises TTA forwards and Grad-CAM.
- **Why:** Grad-CAM registers hooks on the shared model; concurrent forwards would interleave them. Upload parsing still runs concurrently.
- **Evidence:** `e41d8bf` (6 concurrent `/predict` calls returned identical predictions).
- **By:** not recorded.

### D7 - Margin and entropy review rules exist but ship OFF (M2)
- **Decision:** `serving/routing.py` supports `low_margin` and `high_entropy`; they activate only when `REVIEW_MARGIN_MIN` / `REVIEW_ENTROPY_MAX` are set.
- **Why:** measured, they do not clear the bar. On the 1,340-image test set, with the threshold taken from the validation split only, the margin rule at its p2 threshold moved tag precision 0.898 -> 0.914 and home recall 0.799 -> 0.784: about one photo of precision bought per photo of recall lost, inside the noise (95% CI half-width about 3 points on 420 tagged photos). Misclassification AUROC on test: confidence 0.857, margin 0.859, entropy 0.852, max-logit 0.838, energy 0.818, Mahalanobis 0.683: no cheap score beats the existing confidence. The roadmap's two target cases (`cloister -> bathroom` 0.91, `nursery -> greenhouse` 0.99) have margins 0.88 and 0.99, so no margin or entropy threshold can flag them.
- **Rejected:** shipping the rules on by default (a wash), and tuning thresholds on the field set (only 3 wrong tags there; that would be fitting noise).
- **Consequence:** the roadmap's acceptance line "the two confidently-wrong cases route to review" is **not met** by M2. See `reports/review_routing.md`.
- **Evidence:** `evaluation/collect_probs.py` outputs; `tests/test_routing.py` (8 planted-bug mutants, all killed).
- **By:** Claude Sonnet 5.5 (`claude-sonnet-5-5`), session 59024edd, 2026-09-29.

### D8 - Field ground truth is drafted from filenames; photos stay outside the repo (M5)
- **Decision:** `evaluation/field_labels.csv` maps each of the 72 photos to a class, or `ood` for four photos matching no class (`aquarium`, `autopsy-room`, `igloo`, `ship`). The photos are not committed; `--field-root` points at them.
- **Why:** the folder has no label file, but names mirror the class names (all 72 matched, checked by script). Photos stay out per Pranav's instruction.
- **Known weakness:** labels come from the owner's file names, not from independent annotation. Two names are judgement calls (`Artroom` -> `artstudio`, `salon` -> `hairsalon`), and `subway2` is labelled `subway` though the model says `inside_subway`, a genuinely ambiguous pair. Only 26 of the 72 photos are home classes, so this set can guard against regressions but cannot support an accuracy claim such as ">= 90% on home classes".
- **By:** Claude Sonnet 5.5 (`claude-sonnet-5-5`), session 59024edd, 2026-09-29.

### D9 - The home-class list lives in `serving/scope.py`
- **Decision:** moved out of `app.py`; `app.HOME_CLASS_LABELS` is unchanged for callers.
- **Why:** the scoped retrain, the field evaluation and the API must agree on what "home" means, and only the API could import it without loading Flask and a model.
- **By:** Claude Sonnet 5.5 (`claude-sonnet-5-5`), session 59024edd, 2026-09-29.

### D10 - The scoped 24 + other model (M3) is NOT shipped
- **Decision:** `training/train_scoped.py` and `models/scoped_best.pth` (local, git-ignored) stay as an experiment; the served model remains the 67-class `phase2_ema.pth`. No release, manifest or serving change.
- **What was built:** the served backbone and trunk with a fresh 512 -> 25 last layer (24 home classes + `other`), 15 epochs, same seed-42 validation split, temperature 0.674 and threshold 0.641 fitted on that split only. Best 25-way validation accuracy 90.86% (single crop).
- **Measured** (`reports/scoped_compare.md`, identical metric definitions for both models):

  | | Tag precision | Home recall | Wrong tags |
  |---|---|---|---|
  | MIT test n=1,340: served / scoped | 0.898 / 0.913 | 0.799 / 0.708 | 43 of 420 / 32 of 366 |
  | Field n=72: served / scoped | 0.875 / 0.900 | 0.808 / 0.692 | 3 of 24 / 2 of 20 |

  Precision gain about 1.5 points, inside the noise (test 95% CIs 0.865 to 0.923 vs 0.879 to 0.937); recall loss about 9 points, which is not noise (43 fewer right tags, 377 -> 334, on 472 home photos).
- **The roadmap's two target cases are not fixed:** `cloister -> bathroom` is still auto-tagged at 0.92 and `nursery -> greenhouse` at 0.97. `bar.jpg` and `lobby2.jpg`, true home rooms, are sent to `other`. It does turn `igloo -> wine cellar` (0.85 served) into a review (0.61 < 0.64), a single photo.
- **Why it does not help:** not investigated. The evidence is only that the retrained head, on the same backbone, keeps both confident errors while losing recall.
- **Rejected:** shipping it for the precision gain (a recall loss this size means a real user reviews far more photos for a difference we cannot distinguish from chance).
- **Would change this:** more labelled home-class data, or a stronger backbone (M7), tested on a bigger field set. Roadmap acceptance ">= 90% home-class top-1 on the field set" is also unmeasurable here: the served model scores 0.846 on 26 home photos (95% interval roughly 0.66 to 0.94).
- **By:** Claude Sonnet 5.5 (`claude-sonnet-5-5`), session 59024edd, 2026-09-29.

### D11 - `/predict_batch` runs one batched forward pass per request (M10)
- **Decision:** validate every file first, then stack the TTA views of all valid images into one forward pass (`inference/tta.py: tta_predict_batch`, `single_predict_batch`); results go back into their upload slots. `/predict` keeps the per-image path because it also runs Grad-CAM. Each pass is bounded to 8 images (56 tensors), matching the frontend's chunk size, so a request cannot allocate unbounded activations.
- **Why:** the per-image loop ran 3 small forward passes per photo. Measured on the real model (RTX 4060, 16 field photos, TTA on): 3.89 s -> 1.56 s, **2.49x faster**, identical predicted class on all 16, largest probability difference 0.00045 (float16 autocast noise). CPU-only containers should gain less; that was not measured.
- **Guard:** `tests/test_tta_batch.py` proves batched == per-image on a small BatchNorm network (tolerance 1e-5), plus order, chunking and empty-input cases; endpoint tests check one model call, upload order preserved with an invalid file in the middle, an all-invalid batch never running the model, and TTA on/off selecting the right function. Existing contract tests now fake the batch function instead of the per-image one; what they assert is unchanged.
- **Consequence:** any future change to `tta_predict` (views, weights of the views) must be mirrored in `tta_predict_batch`; the equivalence test fails if they drift.
- **By:** Claude Sonnet 5.5 (`claude-sonnet-5-5`), session 59024edd, 2026-09-29.

### D12 - Roadmap items deliberately not built (non-deployment)
- **Done:** F6 (keyboard: `j`/`k` between photos, `c` to the correction dropdown, verified with real key presses in a browser) and F7 (a tooltip on the confidence figure stating the live calibrated threshold).
- **B10 API keys - not built.** Enforcing a key on `/predict*` would break the bundled same-origin UI, because a browser page cannot hold a secret. Making it useful needs a UI login or a separate keyed API surface: a product decision, not a coding one.
- **M9 ONNX + INT8 - not built.** It adds `onnxruntime` to the serving image (CONSTRAINTS "ask first") and its benefit is CPU latency, which was not measured on the target host.
- **B9 async job API - not built.** The frontend already sends chunks of 8 and paints incremental progress (F1); a job queue adds state without a measured need. P2 in the roadmap.
- **M4 (listing-photo fine-tune), M7 (stronger backbone), M8 (CLIP zero-shot):** need data or models this repo does not have; D10 shows retraining a head on the current backbone does not move the errors that matter. M6 (conformal review guarantees) is unscoped.
- **By:** Claude Sonnet 5.5 (`claude-sonnet-5-5`), session 59024edd, 2026-09-29.
