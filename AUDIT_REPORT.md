# Audit Report — Explainable Hybrid CV System

**Date:** 2026-07-07 · **Auditor:** Claude (Fable 5) · **Method:** full source read + live execution against saved artifacts. Every number below was measured this session with the project venv (`venv\Scripts\python.exe`, Python 3.10.11, torch 2.5.1+cu121); nothing is quoted from prior docs.

---

## 1. Verified current state (real numbers)

### Environment
| Check | Result |
|---|---|
| GPU | NVIDIA RTX 4060 Laptop, 8 GB (nvidia-smi OK, driver 610.47, CUDA 13.3) |
| `venv` Python 3.10.11 / torch 2.5.1+cu121 | `cuda_available: True` ✅ |
| System Python 3.13.1 / torch 2.10.0+cpu | `cuda_available: False` |
| Last baseline training run | **Ran on the CPU-only system Python 3.13** (log shows `AppData\Roaming\Python\Python313` + "CUDA is not available" warnings) |

**The CPU-vs-GPU discrepancy is resolved:** the machine has a working GPU and the venv can use it; prior runs simply invoked the wrong interpreter. `run_all.bat` activates the venv, but manual runs (per `results/phase1_tail.txt`) used `python` = system Python.

### Model accuracy (measured now, MIT Indoor 67 test set, n=1,340)
| Model | Artifact | Top-1 | Top-5 | Notes |
|---|---|---|---|---|
| Baseline CNN (ResNet-18) | `models/baseline.pth` | **70.82%** | **91.57%** | Loaded into its true architecture (`train_baseline.build_model`); strict load OK |
| Hybrid HOG-SVM | `models/hybrid_svm.pkl` | **10.75%** | — | The *original broken-state* number. The "fixed to 35–45%" claim is false |
| Fusion (Architecture B) | `models/fusion_best.pth` | **does not exist** | — | Never trained |
| Phase 2 ResNet-50 Places365 | `models/phase2_best.pth` | **does not exist** | — | `results/phase2_train.log` is 0 bytes — training never ran |

### Calibration & robustness (baseline ResNet-18, measured now)
- **ECE (uncalibrated, 15 bins): 12.65%** — model is **underconfident** (mean confidence 58.2% vs 70.8% accuracy), consistent with label smoothing 0.1 + MixUp/CutMix. Temperature scaling was never fitted (`models/temperature_cnn.json` absent), so `/predict` confidence is uncalibrated.
- Robustness: clean 70.82% → Gaussian noise **54.70%** → blur **62.76%** → occlusion **69.93%** (top-1).
- Top confused pairs: bookstore→library (6), deli→bakery (6), videostore→library (6), library→bookstore (5), artstudio→office (4).

### The Flask app is 100% non-functional
Importing `app.py` (exactly what `python app.py` does) shows **every model path fails to load**:
1. Baseline: `UnboundLocalError: local variable 'CNNBaseline' referenced before assignment` — the `from models.cnn_baseline import CNNBaseline` and `import joblib` statements *inside* the fusion branch of `load_models()` make those names function-local, killing lines 141/171 before any checkpoint I/O.
2. Even past that, `baseline.pth` (raw **ResNet-18** state_dict) cannot load into `CNNBaseline` (**ResNet-50** Places365): `size mismatch for model.layer1.0.conv1.weight: [64,64,3,3] vs [64,64,1,1]` (reproduced).
3. Hybrid: same `joblib` shadowing → `UnboundLocalError`.
4. Fusion: checkpoint missing **and** `load_models()` never declares `global fusion_model, fusion_cnn, fusion_pca`, so they'd stay `None` even if it existed.

Net: every `/predict` currently returns 503 (or silently degrades). The same ResNet-18-into-ResNet-50 mismatch also breaks `evaluation/evaluate_models.py` (`evaluate_baseline`), `evaluation/robustness_test.py`, and `explainability/gradcam_explain.py`.

### Data pipeline
- Dataset is balanced and complete: train 5,360 imgs (77–83/class), test 1,340 (17–23/class), 67 classes. No class below the 50-image threshold.
- **PCA(512) on HOG features explains only 67.3% of variance** (both `data/hog_pca_model.pkl` and the SVM pipeline's internal PCA). The in-code claim "512 retains ~95% variance" is false; the prior agent's notes quietly redefined the target to ">65%".
- **Train/serve feature skew (new bug):** serving resizes RAW→224 (LANCZOS)→128 before feature extraction; training resizes RAW→128 directly. Measured max feature deviation 0.239 on the same image. LBP dtype is consistent (both uint8) — this is a *resize-path* skew, not the old dtype bug.

---

## 2. Confirmed-fixed issues (with evidence)

| Claim | Verdict | Evidence |
|---|---|---|
| Val split determinism (no shared mutated dataset) | ✅ Fixed in `train_baseline.py` and `train_phase2.py` | Both build two separate `MITIndoorDataset` instances + `Subset` over a seed-42 permutation; `run_calibration.py` re-derives the same split. ⚠️ But `train_fusion.py` "validates" on the **test set** (see §4). |
| LBP computed on uint8 | ✅ Fixed in both paths | `extract_hog_features.py:133` (with assert) and `app.py:97` |
| No deprecated `torch.cuda.amp` | ✅ | Repo-wide grep: zero hits; all use `torch.amp.*` |
| No `pretrained=True` | ⚠️ One remains | `cnn_baseline.py:19` — inside the (dead, broken) `torch.hub` Places365 branch |
| Places365 weights load | ✅ via local file | `CNNBaseline(67, 'resnet50_places365_local')` loads `resnet50_places365_weights.pth.tar` successfully (verified by instantiation). The `torch.hub` branch is still broken but unused. |
| Checkpoint loader used everywhere | ❌ False | Inline `torch.load`/`load_state_dict` in `app.py` (fusion), `evaluate_models.py` (×4), `demo/run_demo.py`, `cnn_baseline.py` — 5 duplicate implementations besides `utils/checkpoint.py` |
| True fusion (joint training) | ❌ Not real yet | `HybridFusion` module is a genuine joint architecture, but it was **never trained**, and Stage B (joint fine-tune) in `train_fusion.py:111-123` is a stub that prints "Model configuration ready for full training." and exits |

---

## 3. Issues believed fixed but NOT verified / regressed

1. **Hybrid SVM accuracy** — still **10.75%** measured. The `.pkl` is newer than the regenerated features, so either it's the stale tracked artifact or retraining on the improved features genuinely doesn't help; root-causing is backlog item B-8.
2. **"Phase 2 training started/complete"** — never happened. 0-byte log, no checkpoint. The prior agent's Phase-4 report declared "GO for Phase 5" without verification.
3. **Calibration** — code exists and is wired into `app.py`/`tta.py`, but no scaler was ever fitted. The prior agent's own synthetic test hit the optimizer bound (T=10.0, the upper clamp) and *increased* ECE — a red flag that was recorded and ignored.
4. **TTA** — implemented (`inference/tta.py`) and wired into `/predict`, but gated on `?tta=1` **query param of a POST route**; the frontend never sends it. Effectively dead code in serving.
5. **EMA** — wired in `train_phase2.py` but depends on `timm`, which is **not installed** in the venv → would print a warning and silently disable.

---

## 4. Newly discovered bugs (in no prior doc)

| # | Bug | Location |
|---|---|---|
| N1 | Shadowed `CNNBaseline`/`joblib` imports → `UnboundLocalError` → **zero models load in the app** | `app.py` `load_models()` |
| N2 | Missing `global fusion_model, fusion_cnn, fusion_pca` — fusion could never be served | `app.py:131` |
| N3 | `baseline.pth` (ResNet-18) loaded into ResNet-50 `CNNBaseline` in 4 consumers → crash | `app.py`, `evaluate_models.py`, `robustness_test.py`, `gradcam_explain.py` |
| N4 | `utils/checkpoint.load_checkpoint` cannot read the phase-2 format `{'model_state': …, 'ema_state': …}` → the whole phase2→embeddings→fusion→calibration chain fails at the seam (reproduced with synthetic checkpoint) | `utils/checkpoint.py:32` |
| N5 | `load_checkpoint(..., load_ema=True)` — parameter doesn't exist → `TypeError` when phase2 checkpoint appears | `evaluate_models.py:263` |
| N6 | Fusion trainer uses the **test set** as validation and saves `fusion_best.pth` by test accuracy — leakage | `train_fusion.py:77` |
| N7 | Stage B joint fine-tuning is a stub (optimizer built, no training loop) | `train_fusion.py:111-123` |
| N8 | Train/serve resize skew: RAW→224→128 in serving vs RAW→128 in training (max feature diff 0.239) | `app.py:66` vs `extract_hog_features.py:176` |
| N9 | Softmax over `LinearSVC.decision_function` presented as "calibrated confidence" — mathematically invalid; also feeds the 0.30 out-of-scope threshold, making rejection behavior meaningless for the hybrid arm | `app.py:232` |
| N10 | `requirements.txt` missing `Flask` and `flask-cors` (installed ad hoc, unpinned) | `requirements.txt` |
| N11 | `load_state_dict(strict=False)` in the shared loader silently accepts partial loads — a wrong-architecture checkpoint with coincidentally matching key names would "load" | `utils/checkpoint.py:54` |
| N12 | `extract_cnn_embeddings.py` fallback path points at `models/baseline_best.pth`, which doesn't exist; with no checkpoint it proceeds with **random weights** after a warning | `extract_cnn_embeddings.py:16` |
| N13 | `train_fusion.py` uses backbone `'resnet50_places365'` (broken hub variant) instead of `'resnet50_places365_local'` | `train_fusion.py:14` |
| N14 | Raw `traceback.format_exc()` returned to HTTP clients | `app.py:404` |
| N15 | README contains leftover instruction fragments ("STEP 3: Save the file…"); `results.txt` is a stale broken-era report | `README.md:121` |
| N16 | ~3 months of work uncommitted: 660 insertions/408 deletions across 12 tracked files + ~20 untracked files; last commit 2026-04-08 | git status |

---

## 5. Prioritized backlog

Effort: S < ½ day · M = ½–2 days · L > 2 days. Impact on the "portfolio-defining" goal.

| ID | Item | Root cause | Impact | Effort | Priority |
|---|---|---|---|---|---|
| B-1 | Fix `app.py` model loading (N1, N2, N3: shadowed imports, arch registry per checkpoint, fusion globals) | Copy-paste imports; no arch metadata in checkpoints | App goes from 0 working models → serving | S | **P0** |
| B-2 | Unify checkpoint I/O: one loader that understands raw/`state_dict`/`model_state`/`ema_state`, strict-by-default, `load_ema` param; retrofit 5 inline call sites (N4, N5, N11) | Format drift between phases | Unblocks entire phase-2 chain; kills a bug class | M | **P0** |
| B-3 | Environment pinning: everything runs via `venv`; add Flask/flask-cors to pinned `requirements.txt`; device banner in every entry point (N10) | Two interpreters on PATH | Prevents future silent CPU runs | S | **P0** |
| B-4 | Commit current work; tag audit baseline (N16) | Process | Protects 3 months of work | S | **P0** |
| B-5 | Run `train_phase2.py` on GPU (install `timm` for EMA) → real ResNet-50-Places365 checkpoint | Never executed | Expected ~78–85% top-1 (vs 70.8) — the single biggest accuracy jump | M (60–90 min GPU, mostly unattended) | **P1** |
| B-6 | Fit temperature scaling on the val split; report ECE before/after + reliability diagram; wire into `/predict` (already plumbed) | Never run | Production-ML signal; fixes confidence UX | S (after B-5) | **P1** |
| B-7 | Fusion for real: extract embeddings → build fusion dataset → Stage A → implement true Stage B image-level joint fine-tune; carve a proper val split (fix N6, N7, N13) | Stub + leakage | Makes "hybrid" claim honest; ablation-ready | L | **P1** |
| B-8 | Root-cause the 10.75% HOG-SVM (or retire the arm in favor of the already-coded `train_embedding_svm.py` with `CalibratedClassifierCV`); fix serve-path resize skew (N8) and invalid softmax-confidence (N9) | Multiple | Honesty of the "hybrid" story | M | **P1** |
| B-9 | pytest suite: split determinism, checkpoint round-trip, LBP dtype regression, feature-parity (train vs serve), API contract, calibration monotonicity | None exist | Guards every fixed bug structurally | M | **P2** |
| B-10 | Ablation table with real numbers: CNN-only vs fusion; Places365 vs ImageNet; ±TTA; ±EMA | Framework exists (`ablation_study.py` is plan-only) | Highest interview-value per hour | M (mostly GPU time) | **P2** |
| B-11 | Robustness + confusion-matrix writeup (scripts work once B-1/B-2 land; numbers above are the baseline) | Blocked by N3 | Analytical-depth signal | S | **P2** |
| B-12 | Serving hardening: FastAPI migration (or minimum: input validation, size limits, structured errors, `/metrics` latency percentiles, model-version metadata) (N14 + Step 2.9/2.10) | Flask MVP | Production-engineering signal | M | **P2** |
| B-13 | MC-Dropout / small-ensemble uncertainty + OOD rejection driven by calibrated entropy, replacing the arbitrary 0.30 threshold | Missing | Differentiator; feeds headline use case | M | **P3** |
| B-14 | Frontend overhaul (design system, calibrated-confidence badge, explainability tabs, robustness dashboard, batch demo, a11y, mobile) | MVP single-file UI | The part reviewers *see* | L | **P3** |
| B-15 | Headline use case build (see §6) | — | Turns benchmark into product | L | **P3** |
| B-16 | Stretch: CLIP zero-shot ensemble/search, ONNX+INT8 export, Docker packaging, experiment-tracking CSV | — | Nice-to-have signals | M–L | **P4** |

Deliberately **not** doing: backbone upgrade to ConvNeXt/EfficientNet before B-5 proves out the Places365 ResNet-50 (one variable at a time); keeping the HOG-SVM arm alive purely for narrative if B-8 shows it contributes <1–2% in fusion ablation.

### B-8 status (Session 4, 2026-07-12)

- **HOG-SVM arm RETIRED from serving.** Measured 10.75% top-1 (67-class, ~7× chance). Removed the `hybrid` branch, its startup loader, `/health` field, and the dead UI button. The invalid softmax-over-`decision_function` confidence (N9) was **deleted, not patched**.
- **N8 serve-skew fixed** as a single source of truth for the resize path (see commit 2): serving now extracts HOG from the RAW image via `preprocessing/extract_hog_features.extract_features_from_rgb` (RAW→128), identical to training, instead of the RAW→224→128 duplicate.
- **HybridFusion disabled from the default `/predict` path** but code + `fusion_best.pth` + B-7 ablation numbers retained for B-10 and the README (honest finding: fusion 82.01% test < CNN-only 83.21%).
- **B-8a (FUTURE backlog, not built this session):** principled SVM replacement already scaffolded at `training/train_embedding_svm.py` — train a `LinearSVC`/`SVC` on the **CNN embeddings** (not HOG) wrapped in `sklearn.calibration.CalibratedClassifierCV` (Platt/isotonic) to get *real* probabilities instead of softmax-over-decision_function. Only worth doing if a classical-head arm is wanted alongside the CNN; needs its own session (involves training). Logged per user instruction; deliberately not implemented in B-8.

---

## 6. Recommended headline framing (Step 4)

**Real-estate / interior-space photo tagging tool** — bulk-upload a folder of property photos, auto-tag by room type with calibrated confidence, flag low-confidence/OOD images for human review, export CSV.

**Why this one:** it's the only framing where the existing 67-class indoor taxonomy *is* the product (bedroom/kitchen/bathroom/living-room are literally listing categories), so the benchmark work transfers directly instead of needing a new modality. Calibration + uncertainty (B-6/B-13) become a visible product feature ("review these 7 photos") rather than a metrics table, Grad-CAM becomes an auditing tool for mis-tagged listings, and batch mode + CSV export is a compelling, low-risk demo that recruiters can try with any Zillow screenshot. The accessibility describer is more socially resonant but requires a VLM+TTS chain and makes implicit safety promises a 67-class classifier can't back; CLIP search is better as a P4 stretch feature layered onto this same app than as the headline.

---

## 7. GO / NO-GO

**NO-GO for feature upgrades. Stabilization first.**

The system's published claims and its measured reality have fully diverged: the demo app serves nothing, the flagship "hybrid" model doesn't exist, and the only two real artifacts are a decent CPU-trained ResNet-18 (70.8%) and a broken SVM (10.75%). Proceeding to Step 3–5 work on this foundation would build on sand. The P0 block (B-1…B-4) is under a day of work and makes every subsequent claim verifiable; P1 (B-5…B-8) then delivers the real accuracy story on hardware that this audit proved is available.

## 8. Proposed execution order

| Session | Scope | Exit criteria (all verified by running) |
|---|---|---|
| 1 | P0: B-1, B-2, B-3, B-4 | `python app.py` (venv) loads baseline + SVM; `/predict` returns 200 with Grad-CAM; `evaluate_models.py` + `robustness_test.py` run clean; work committed |
| 2 | P1a: B-5, B-6 | `phase2_best.pth` exists with logged val acc; test top-1 reported; ECE before/after + reliability diagram generated; app serves calibrated ResNet-50 |
| 3 | P1b: B-7, B-8 | Fusion trained with honest val split, Stage B real; fusion vs CNN-only ablation numbers; SVM arm fixed or formally retired |
| 4 | P2: B-9, B-10, B-11 | pytest green in CI-able form; ablation table + confusion/robustness writeup in `results/` |
| 5 | P2/P3: B-12, then B-14 start | Hardened API with `/health`+`/metrics`; frontend redesign underway |
| 6 | P3: B-14 finish, B-15 | Headline demo (bulk tagging + CSV + review queue) end-to-end |
| 7+ | P4 stretch (B-16) | As approved |
