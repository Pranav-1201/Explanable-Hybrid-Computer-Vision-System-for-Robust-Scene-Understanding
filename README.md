# Explainable Hybrid Vision for Robust Scene Understanding

![docker-smoke](https://github.com/Pranav-1201/Explanable-Hybrid-Computer-Vision-System-for-Robust-Scene-Understanding/actions/workflows/docker-smoke.yml/badge.svg)

This project implements an explainable hybrid computer vision system that combines deep learning with handcrafted features to improve scene understanding. The goal is to achieve higher robustness and interpretability compared to standard CNN-based approaches.

---

## 📊 Model status & honest results (MIT Indoor 67 test set, n = 1,340)

The **served** model is CNN-only — a **ResNet-50 (Places365) fine-tune**, the honest best performer:

| Arm | What it is | Test top-1 | Status |
|---|---|---|---|
| **CNN (served)** | ResNet-50 Places365, EMA weights, temperature-calibrated, TTA default | **83.88%** (83.21% single-crop; top-5 97.84%) | **Served** |
| Fusion (Architecture B) | HybridFusion: CNN 2048 + HOG-PCA 512, jointly fine-tuned | 82.01% | Disabled research finding |
| HOG-SVM | HOG spatial pyramid → LinearSVC | 10.75% | Retired |

**Honest negative result (kept as-is).** The true feature-fusion hybrid (Architecture B) was trained for real — two-stage: frozen-backbone Stage A → joint fine-tune Stage B — and **does not beat CNN-only**. It won on validation (84.14% vs 83.02%) but that did **not** generalize: on the held-out test set it is **−1.2%** (82.01% vs 83.21%), with Stage B overfitting (train ≈ 98.8%). Adding handcrafted HOG features to a strong Places365 CNN did not earn its complexity, so the fusion arm is **disabled from the serving path** but retained as a documented research artifact — `models/hybrid_fusion.py` (with `ablate_hog()`), `training/train_fusion.py`, `models/fusion_best.pth` — for the formal ablation study (B-10). The classical HOG-SVM arm scored 10.75% (≈ 7× chance for 67 classes) and was **retired** entirely; its invalid softmax-over-`decision_function` confidence was removed rather than patched.

Confidence is temperature-scaled (T = 0.5142; ECE 33.37% → 5.57%) with a calibrated out-of-scope rejection threshold.

### Ablation (B-10)

Full table in [`reports/ablation_table.md`](reports/ablation_table.md), regenerate with `python evaluation/ablation_study.py`. Test set, n = 1,340, inference only — nothing was retrained.

| Configuration | Test top-1 | Test top-5 |
|---|---|---|
| ResNet-50 Places365, raw weights | 77.91% | 94.33% |
| ResNet-50 Places365, EMA | 83.21% | 97.76% |
| **ResNet-50 Places365, EMA + TTA — served** | **83.88%** | **97.84%** |
| Fusion (CNN + HOG), full | 82.01% | 96.49% |
| Fusion, HOG branch zeroed | 80.45% | 96.49% |
| ResNet-18 ImageNet (context only) | 70.82% | 91.57% |

Three things this actually shows:

1. **EMA is the single biggest win: +5.30 pts** (77.91 → 83.21) — and this is the component that was silently broken until the EMA decay bug was found and fixed.
2. **HOG is not worthless; the fusion architecture is.** Zeroing the HOG branch costs the fusion head **1.56 pts** (82.01 → 80.45), so the head genuinely uses HOG. Yet full fusion still lands **1.2 pts below plain CNN-only**. HOG adds real signal — the fusion wrapper just costs more than that signal is worth. Both halves of that are reported, not only the flattering one.
3. **TTA is now the served default: +0.67 pts** (83.21 → 83.88), which is why the served row above is the EMA + TTA line. It costs a measured +22 ms/image (7 forward passes vs 1 — imperceptible in the demo), and is opt-out per request with `tta=0`.

Two caveats that change how the table reads: the HOG row is a **test-time** ablation on a head that *was trained with* HOG, not a retrained no-HOG control. The ResNet-18 row is **context, not a controlled ablation** — it differs in architecture *and* recipe, so it does not isolate Places365-vs-ImageNet pretraining.

---

## 🔍 Key Features
- Hybrid architecture combining CNN features with handcrafted features (HOG, texture, edges, corners)
- Modular preprocessing and feature extraction pipeline
- Baseline CNN and hybrid model implementations
- Robust training and evaluation framework
- Model robustness testing under perturbations
- Explainability using Grad-CAM visualizations
- Clean, reproducible, and well-structured codebase

---

## 🧠 Project Pipeline
1. **Data Preparation & Preprocessing**
   - Dataset loading and cleaning
   - Image preprocessing
   - Feature precomputation

2. **Handcrafted Feature Extraction**
   - HOG features
   - Texture, edge, and corner descriptors

3. **Deep Learning Models**
   - Baseline CNN
   - Hybrid CNN combining CNN and handcrafted features

4. **Training**
   - Baseline model training
   - Feature-based and hybrid model training

5. **Evaluation & Robustness Testing**
   - Accuracy and confusion matrix evaluation
   - Robustness testing under noise and perturbations

6. **Explainability**
   - Grad-CAM visualizations to interpret CNN decisions

---

## 📂 Project Structure
CV+DLPROJECT/
├── preprocessing/ # Data preprocessing and feature extraction
├── data/ # Dataset loaders and preparation scripts
├── classical_features/ # Handcrafted feature extraction modules
├── models/ # CNN and hybrid model definitions
├── training/ # Training scripts
├── evaluation/ # Evaluation and robustness testing
├── explainability/ # Grad-CAM explainability tools
├── utils/ # Utility functions
├── demo/ # Demo scripts
├── requirements.txt # Python dependencies
├── run_all.bat # End-to-end execution script
└── README.md


---

## 📊 Dataset
This project uses the **MIT Indoor Scene Recognition dataset**.

Due to size constraints, the dataset and extracted feature files are **not included** in this repository.

Please download the dataset separately and place it in the appropriate data directory before running the pipeline.

---

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

---

## ⚙️ Installation
1. Create a virtual environment
```bash
python -m venv venv

Activate the environment

Windows:

venv\Scripts\activate

Linux / macOS:

source venv/bin/activate

Install dependencies

pip install -r requirements.txt
▶️ Running the Project

To run the complete pipeline:

run_all.bat

Individual stages (preprocessing, training, evaluation, explainability) can also be run independently using the respective scripts.

🧪 Explainability Example

Grad-CAM is used to visualize which regions of an image influence the CNN’s predictions, improving model interpretability and trust.

🚀 Motivation

Standard CNNs often act as black boxes and can be sensitive to distribution shifts. By fusing handcrafted features with deep representations and adding explainability, this project aims to build a more robust and interpretable vision system.

📌 Notes

Model weights and large feature files are intentionally excluded from version control.

All experiments are fully reproducible using the provided scripts.

👤 Author

Pranav Upadhyay