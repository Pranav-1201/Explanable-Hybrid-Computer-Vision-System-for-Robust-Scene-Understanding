# Explainable Hybrid Vision for Robust Scene Understanding

This project implements an explainable hybrid computer vision system that combines deep learning with handcrafted features to improve scene understanding. The goal is to achieve higher robustness and interpretability compared to standard CNN-based approaches.

---

## 📊 Model status & honest results (MIT Indoor 67 test set, n = 1,340)

The **served** model is CNN-only — a **ResNet-50 (Places365) fine-tune**, the honest best performer:

| Arm | What it is | Test top-1 | Status |
|---|---|---|---|
| **CNN (served)** | ResNet-50 Places365, EMA weights, temperature-calibrated | **83.21%** (top-5 97.76%) | **Served** |
| Fusion (Architecture B) | HybridFusion: CNN 2048 + HOG-PCA 512, jointly fine-tuned | 82.01% | Disabled research finding |
| HOG-SVM | HOG spatial pyramid → LinearSVC | 10.75% | Retired |

**Honest negative result (kept as-is).** The true feature-fusion hybrid (Architecture B) was trained for real — two-stage: frozen-backbone Stage A → joint fine-tune Stage B — and **does not beat CNN-only**. It won on validation (84.14% vs 83.02%) but that did **not** generalize: on the held-out test set it is **−1.2%** (82.01% vs 83.21%), with Stage B overfitting (train ≈ 98.8%). Adding handcrafted HOG features to a strong Places365 CNN did not earn its complexity, so the fusion arm is **disabled from the serving path** but retained as a documented research artifact — `models/hybrid_fusion.py` (with `ablate_hog()`), `training/train_fusion.py`, `models/fusion_best.pth` — for the formal ablation study (B-10). The classical HOG-SVM arm scored 10.75% (≈ 7× chance for 67 classes) and was **retired** entirely; its invalid softmax-over-`decision_function` confidence was removed rather than patched.

Confidence is temperature-scaled (T = 0.5142; ECE 33.37% → 5.57%) with a calibrated out-of-scope rejection threshold.

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



---


## STEP 3: Save the file
Save `README.md`.


---


## STEP 4: Check Git status
Back in your terminal:


```bash
git status