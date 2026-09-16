# FAANG-Level Audit & Deployment Roadmap

**Date:** 2026-08-07 · **Auditor:** Claude (Fable 5) · **Scope:** full source read of `D:\CV+DLPROJECT` (75 tracked files), field-test results from two real-world CSV runs, and the external project folder on E:\. Audit + planning only — no code was changed.

---

## PHASE 1 — FULL COMPREHENSION

### Tech stack (detected)
| Layer | What it is |
|---|---|
| Language | Python 3.10 (venv), single-file vanilla-JS frontend |
| ML | PyTorch 2.5.1+cu121, torchvision 0.20.1, timm 1.0.27 (EMA), grad-cam 1.5.5, ttach, scikit-learn 1.7.2, scikit-image, OpenCV |
| Serving | Flask 3.1.3 + flask-cors 6.0.2, dev server (`app.run`, app.py:549) |
| Frontend | One 605-line `frontend/index.html` — no framework, no build step, design tokens, dark mode, a11y live regions |
| Data | MIT Indoor 67 (5,360 train / 1,340 test), folder-per-class; gitignored |
| Build/run | `run.bat` / `run.ps1` venv guards, `run_all.bat` pipeline |
| CI/CD | **None** (no .github/, no Dockerfile, no deploy config) |

### Architecture map
- **Entry point:** `app.py` — loads model at import (`load_models()`, app.py:236). Serves `frontend/index.html` at `/` (same origin).
- **Served model:** ResNet-50 (Places365 weights, local .tar) with custom 2-layer head (`models/cnn_baseline.py:52-59`), EMA weights from `models/phase2_best.pth`, temperature-calibrated (T=0.5142), TTA (7 forward passes) on by default (`inference/tta.py`).
- **Endpoints:** `GET /health`, `GET /classes`, `POST /predict` (single, Grad-CAM), `POST /predict_batch` (≤50 images, review-queue routing).
- **Review routing:** `low_confidence` if calibrated max-prob < 0.478 (`models/calibration_config.json`); `out_of_scope` if predicted class ∉ 24-entry `HOME_CLASS_LABELS` (app.py:131-156).
- **Data flow (frontend→backend):** drag-drop → single multipart POST to `/predict_batch` → JSON per-image results → card grid + CSV export; Grad-CAM on demand via `/predict`.
- **Research arms (not served, honest negative results):** HybridFusion CNN+HOG (82.01% < CNN-only 83.21%, disabled), HOG-SVM (10.75%, retired).

### Dependencies
Fully pinned (`pip freeze`), current versions. Only flags: torch CUDA wheels need the PyTorch index (documented in requirements.txt header); `ttach` 0.0.3 is unmaintained but tiny.

### Tests & CI
- 5-file pytest regression suite (`tests/`) guarding the historical bug classes: split determinism, checkpoint round-trip, feature parity (train/serve skew), calibration monotonicity. Good targeted coverage; **no API contract tests, no CI runner, no coverage metric, no frontend tests.**

### Deployment setup
**None.** Additionally two hard deployment blockers exist in code:
1. `discover_classes()` (app.py:166) reads class names from `data/MIT_Indoor/train` — the server literally cannot start correctly without the 6,700-image dataset on disk.
2. All checkpoints are gitignored (`*.pth`) — a fresh clone cannot serve anything; there is no artifact download path.

### Field-test results analysis (the two CSVs)
- **Run 1 (50 MIT-style web images):** ~40/47 decodable top-1 correct ≈ **85%** — consistent with the benchmarked 83.9%. The model is not lying about its accuracy.
- **Run 2 (10 novel phone-style photos):** 4/5 home-room photos tagged correctly (bathroom .99, closet .91, kitchen .92, garage .67); `bar.jpg` misclassified but **correctly routed to review** (conf 0.28); both true-OOD images (aquarium, ship) correctly rejected.
- **Failure mode #1 — format, not model:** 5 of 60 images returned `invalid` solely because they are **AVIF** (`ALLOWED_FORMATS`, app.py:100, and Pillow lacks an AVIF/HEIC plugin). Phones default to HEIC; this is the single cheapest accuracy win available.
- **Failure mode #2 — confidently wrong in-scope:** `cloister→bathroom @ 0.91`, `nursery→greenhouse @ 0.99`. T=0.514 **sharpens** probabilities, so semantic near-OOD errors sail past the max-prob threshold. Max-softmax alone cannot catch these; need margin/energy signals or a scoped retrain.
- **Failure mode #3 — correct but rejected:** `office2 @ 0.2519` predicted the right class but fell under threshold. Domain shift (modern photos vs 2009-era MIT Indoor) depresses confidence.

**Conclusion:** accuracy concern is real but narrower than it looks — the model performs at benchmark on in-distribution images; the perceived gap comes from (a) format rejects, (b) domain shift on modern photos, (c) sharpened overconfidence on near-OOD.

---

## PHASE 2 — FAANG-STYLE GRADES

| Category | Score | Evidence |
|---|---|---|
| Correctness & functionality | **7** | Works end-to-end; field CSVs confirm benchmark. Deductions: AVIF/HEIC rejects (app.py:100); confidently-wrong in-scope (T=0.514 sharpening + max-prob-only routing, app.py:508-511); batch body-limit bug below. |
| Architecture & code quality | **6.5** | Clean module split, single source of truth for features (`extract_features_from_rgb`), unified checkpoint loader (`utils/checkpoint.py`), honest inline docs. Deductions: module-level globals + `load_models()` at import (app.py:55-58, 236); scratch scripts in root (`verify2.py`, `debug_hybrid.py`, `classesPrint.py`); serving and research code interleaved. |
| Backend logic | **6** | Strong upload validation (app.py:240-291), correlation-ID errors (app.py:437), per-file batch fault isolation. Deductions: **`MAX_CONTENT_LENGTH=10MB` applies to the whole request** (app.py:106) while the UI promises "up to 50 photos" (index.html:292) — 50 real photos 413s; sequential batch × TTA×7 = up to 350 forward passes per request with no progress; classes coupled to dataset dir (app.py:166). |
| Frontend/UI | **7.5** | Genuinely polished: token system, dark mode, RFC-4180 CSV with BOM (index.html:441-467), object-URL cache with revocation. Deductions: single monolithic request (no chunking/progress), whole-grid error state on batch failure (index.html:483). |
| UX | **7** | Review-queue concept is the product's best idea; filter + CSV workflow coherent. Deductions: HEIC/AVIF phone photos silently "invalid"; no way to correct a tag in-app (review queue has no *action*); no per-file progress. |
| Security | **6** | Decompression-bomb guard, decode-verify, no tracebacks to clients, `weights_only=True` checkpoint loads. Deductions: CORS wide-open default (app.py:35), no rate limiting, no auth story, no CSP on the served page. Acceptable for demo, not for public URL. |
| Performance | **5.5** | TTA×7 default per image; batch is fully sequential per image (app.py:484-501) — no cross-image tensor batching; `/predict` echoes the original image back as base64 (app.py:429) that the client already has; GradCAM object rebuilt per request (app.py:313). No ONNX/quantization. |
| Test coverage & reliability | **6** | Targeted regression suite for every historical bug class is genuinely good practice. No CI, no API tests, no eval-regression gate (nothing stops accuracy from silently dropping). |
| Documentation & DX | **7.5** | README with honest negative results and real ablations (rare and valuable); AUDIT_REPORT.md is exemplary. Missing: API reference, weights-acquisition instructions for a fresh clone, LICENSE. |
| Deployment readiness | **2** | Flask dev server (app.py:549), no Dockerfile, no CI/CD, weights unobtainable from a clone, dataset-dependent startup, no metrics/monitoring/rollback. |

**Overall: 6.1/10 — a strong, honest MVP.** Well above prototype: it runs end-to-end, its claims are measured, its worst instincts (fake hybrid, invalid confidence) were already excised, and the frontend is portfolio-quality. It is **not production-grade**, and the gap is concentrated almost entirely in deployment readiness and serving performance — which is fixable in days, not months.

---

## PHASE 3 — IMPROVEMENT BRAINSTORM (Impact × Effort)

### ML / accuracy
| # | Idea | Why | I×E |
|---|---|---|---|
| M1 | Add HEIC/AVIF decode (`pillow-heif`, `pillow-avif-plugin`) | 8% of field images rejected on format alone; phones default HEIC | **H×L** |
| M2 | Margin + entropy review routing (top1−top2 gap, predictive entropy) alongside max-prob | Catches confidently-wrong near-OOD (cloister→bathroom @0.91) without retraining | **H×L** |
| M3 | Retrain a **scoped head**: 24 home classes + "other" (grouping the 43 non-home classes), from the same Places365 backbone | The product is 24-class; training the actual task raises both accuracy and confidence semantics; "other" class gives principled out-of-scope | **H×M** |
| M4 | Fine-tune on modern real-estate photos (e.g. Kaggle House Rooms, scraped listing sets) | Closes the 2009-web-image → modern-wide-angle-HDR domain gap seen in run 2 | H×M |
| M5 | Build a labeled **field eval set** from the user's two test folders + future uploads; wire as regression benchmark | Turns ad-hoc CSV eyeballing into a tracked metric; gates deploys | **H×L** |
| M6 | Re-examine calibration: T=0.514 sharpens; consider per-bin check + conformal prediction for review routing | Principled "review these N photos" guarantee (e.g. 95% coverage) — strong differentiator | M×M |
| M7 | Modern backbone (ConvNeXt-T / EfficientNetV2 / SigLIP fine-tune) | MIT Indoor SOTA is 90%+; +4-6 pts available | M×M |
| M8 | CLIP/SigLIP zero-shot arm for arbitrary user-defined tags | Generalizes product beyond fixed taxonomy; also a free second opinion for OOD | M×M |
| M9 | ONNX export + INT8 quantization for CPU serving | 2-4× CPU latency win; enables free-tier hosting | M×M |
| M10 | Cross-image tensor batching in `/predict_batch` | 50 sequential images → few batched forwards; 5-10× batch latency win | H×M |

### Backend
| # | Idea | Why | I×E |
|---|---|---|---|
| B1 | Ship `classes.json` artifact; remove dataset dependency at serve time | Deployment blocker | **H×L** |
| B2 | Host weights on Hugging Face Hub; startup downloader with checksum | Fresh clone / container can serve | **H×L** |
| B3 | Fix batch body limit: per-file 10MB, request cap ~200MB or client-side chunked uploads (e.g. 5 at a time) | UI promise currently broken for real photos | **H×L** |
| B4 | Production WSGI: gunicorn (Linux container) / waitress; GradCAM behind a lock (backward pass mutates shared model state under threads) | Dev server is single-threaded and unsupported for prod | **H×L** |
| B5 | Dockerfile (CPU) + docker-compose; model + classes baked or volume-mounted | The unit of deployment | **H×L** |
| B6 | Drop base64 `original_image` echo from `/predict` | Payload bloat, client already has the file | M×L |
| B7 | Rate limiting (flask-limiter) + tightened CORS + basic security headers | Public URL hygiene | M×L |
| B8 | `/metrics`: request counts, latency percentiles, review-rate; structured JSON logs | Observability; review-rate is the product KPI | M×M |
| B9 | Async batch with progress (job id + polling or SSE) | UX for 50-photo batches on CPU | M×M |
| B10 | API key auth tier for programmatic `/predict_batch` | Turns demo into usable API | M×M |
| B11 | Move scratch scripts (`verify2.py`, `debug_hybrid.py`, `classesPrint.py`) to `scripts/`; split serving from research code | Repo hygiene, reviewer signal | L×L |

### Frontend / UX
| # | Idea | Why | I×E |
|---|---|---|---|
| F1 | Chunked upload with per-chunk progress ("12/50 tagged…") | Pairs with B3/B9; biggest perceived-speed win | **H×M** |
| F2 | **Actionable review queue**: dropdown to correct/confirm a tag; corrections included in CSV export | Closes the human-in-the-loop loop — the product's whole thesis; also yields labeled data for M4/M5 | **H×M** |
| F3 | Clear message for HEIC/AVIF (until M1 lands: "convert or enable") | Kills the most confusing failure | H×L |
| F4 | Folder drag-drop (`webkitdirectory`) + recursive file picking | Real workflow is "drop the shoot folder" | M×L |
| F5 | Per-card retry for failed files | Fault isolation in UI, not just API | M×L |
| F6 | Keyboard navigation through review queue (j/k, arrow keys) | Reviewer throughput; a11y follow-through | L×L |
| F7 | Confidence tooltip explaining calibration + threshold ("why review?") | Trust; makes the calibration work visible | L×L |

### Testing / CI / Docs
| # | Idea | Why | I×E |
|---|---|---|---|
| T1 | GitHub Actions: pytest + ruff + docker build on PR | Baseline CI | **H×L** |
| T2 | API contract tests (upload validation matrix, batch routing, threshold behavior) | The serving logic is the product; untested today | H×M |
| T3 | Eval-regression job: run field eval set (M5), fail if top-1 or review-precision drops | Prevents silent model regressions | H×M |
| T4 | API docs page (OpenAPI via flask-smorest or hand-written) + LICENSE | DX, portfolio polish | M×L |

---

## STRATEGIC DIRECTION — "what is this actually for?"

Honest framing first: **as a pure classifier, this will never beat commercial vision APIs or a modern VLM.** restb.ai sells exactly this product to MLS platforms; GPT-4V-class models tag rooms zero-shot. So the value cannot be "the model." Three directions:

**Option 1 — Double down: self-hosted property-photo triage tool (RECOMMENDED).**
The durable asset isn't the ResNet — it's the **calibrated review-queue workflow**: batch in, honest confidence out, low-trust photos routed to a human, corrections exported. Real users exist at the small end of the market that commercial APIs ignore: independent listing photographers, small brokerages, and anyone organizing large interior photo archives — people who want a free/local tool, not a per-image API bill, and whose photos never leave their machine (a genuine privacy sell). Ship it as a Docker-runnable web app + hosted demo. Everything already built transfers.

**Option 2 — Generalize: "tag anything with honest confidence."**
Swap the fixed 67-class head for CLIP/SigLIP zero-shot over **user-defined labels**, keep the calibration + review queue + export machinery. Product becomes a general human-in-the-loop image-triage tool (insurance claims, e-commerce catalogs, dataset cleaning). Bigger market, bigger build (~2-3× effort), and it discards the trained-model story. Better as a **v2** on top of Option 1 than a pivot — the review-queue chassis is identical.

**Option 3 — Portfolio freeze.** Deploy the demo, write it up, move on. Zero product ambition; still requires Phase A below.

**Recommendation:** Option 1 now, with F2 (actionable review queue) as the feature that makes it a *tool* rather than a demo; M8/Option 2 as the roadmap's stretch phase. The pitch line: *"An explainable, calibrated interior-photo tagger that knows what it doesn't know — runs anywhere, photos never leave your machine."*

**Deployment target recommendation:**
- **Primary: Hugging Face Spaces (Docker Space, free CPU)** — weights pulled from HF Hub, public demo URL, zero cost, native to an ML portfolio.
- **Secondary (same Dockerfile): Render or Fly.io free/hobby tier** — a "real product" URL with custom domain if wanted.
- One Dockerfile serves both; no cloud lock-in. GPU unnecessary: single-crop CPU inference is ~100-300ms/image, fine once TTA is optional and batching (M10) lands.

---

## PHASE 4 — PRIORITIZED ROADMAP

Priorities: **P0** = blocks deployment or breaks the promise the UI already makes. **P1** = accuracy/trust and product usefulness. **P2** = polish/stretch.

- **P0:** B1, B2, B3, B4, B5, M1, M5, T1, F3
- **P1:** M2, M3, M10, F1, F2, B7, B8, T2, T3, M9
- **P2:** M4, M6, M7, M8, B6, B9, B10, B11, F4-F7, T4

### Phase A — Deployability & correctness fixes (P0, ~2-3 sessions)
**Scope:** `classes.json` artifact + remove dataset dependency (B1); weights to HF Hub + startup download with checksum (B2); batch body-limit fix (B3); waitress/gunicorn + GradCAM lock (B4); Dockerfile (B5); HEIC/AVIF decode (M1); HEIC messaging fallback (F3); repo hygiene (B11).
**Acceptance criteria:** `docker run` on a machine with **no dataset and no pre-placed weights** serves `/` and returns correct predictions; a 50-photo batch of real phone photos (incl. HEIC) completes without 413; pytest green inside the container.
**Verification:** fresh-clone → docker build → run → re-run the user's two field folders through the UI; compare CSVs — zero `invalid` rows for AVIF/HEIC files.

### Phase B — Accuracy & trust (P1-ML, ~2-3 sessions)
**Scope:** labeled field eval set from the two E:\ test folders + ground-truth labels (M5); margin+entropy review routing (M2); scoped 24+other retrain (M3); optional TTA default off with batching (M10).
**Acceptance criteria:** field-set top-1 on home classes ≥ 90% with M3; the two known confidently-wrong cases (cloister, nursery) route to review; review-precision (fraction of reviewed items that were genuinely wrong/OOD) reported.
**Verification:** eval script prints before/after table on the field set + MIT test set; committed to `reports/`.

### Phase C — Backend hardening (P1, ~1-2 sessions)
**Scope:** rate limiting + CORS tighten + security headers (B7); `/metrics` + structured logs (B8); API contract tests (T2); drop b64 echo (B6).
**Acceptance criteria:** contract test matrix green (oversize, wrong format, empty, 51 files, threshold boundary); `/metrics` exposes p50/p95 latency and review-rate.
**Verification:** `pytest tests/ -k api` green; manual `curl` checks documented.

### Phase D — Product UX (P1-frontend, ~2 sessions)
**Scope:** chunked upload + progress (F1); actionable review queue with tag correction + corrected CSV export (F2); folder drop (F4); per-card retry (F5).
**Acceptance criteria:** 50-photo batch shows incremental progress; a reviewer can correct every flagged photo without leaving the page and export includes a `corrected_tag` column.
**Verification:** scripted browser run (Playwright) over a 20-photo fixture folder.

### Phase E — CI & regression safety (P0/P1, ~1 session, can run parallel to B-D)
**Scope:** GitHub Actions: ruff + pytest + docker build (T1); eval-regression job on the field set with accuracy floor (T3); LICENSE + API docs (T4).
**Acceptance criteria:** PR to main runs the full gate; a deliberately broken threshold change fails CI.
**Verification:** green Actions run linked in README.

### Phase F — Deployment (final, ~1-2 sessions)
**Scope:** HF Hub model repo (weights + classes.json + calibration_config.json, versioned); HF Space (Docker) as public demo; optionally Render/Fly from the same image; env config via variables (`MODEL_REPO`, `MODEL_REVISION`, thresholds); uptime monitoring (healthcheck ping) + Sentry (or log tail) for errors.
**CI/CD:** on tag push → Actions builds image → pushes to registry → Space/Render auto-deploys pinned tag.
**Rollback plan:** model and code are versioned independently — `MODEL_REVISION` pins the HF Hub commit; app rollback = redeploy previous image tag (one click on Render / `git revert` on Space). Keep N-1 image and N-1 model revision warm; rollback drill documented in README.
**Acceptance criteria:** public URL serves the demo; a stranger with a phone can drop 20 HEIC photos and get a tagged grid + CSV; rollback drill executed once and documented.
**Verification:** run both field folders against the **deployed** URL; CSVs match local results.

### Sequencing summary
A → E (CI early) → B → C → D → F. Phases B/C/D are independent after A; E gates everything once created. Total: roughly 9-13 working sessions.

### Explicitly deferred (P2 backlog)
Scoped fine-tune on scraped listing data (M4), conformal prediction (M6), modern backbone (M7), CLIP zero-shot custom labels / Option 2 pivot (M8), ONNX+INT8 (M9), async job API (B9), API keys (B10).
