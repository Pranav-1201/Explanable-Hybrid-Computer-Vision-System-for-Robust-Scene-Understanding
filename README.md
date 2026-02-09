# Explainable Hybrid Vision for Robust Scene Understanding

This project implements an explainable hybrid computer vision system that combines deep learning with handcrafted features to improve scene understanding. The goal is to achieve higher robustness and interpretability compared to standard CNN-based approaches.

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