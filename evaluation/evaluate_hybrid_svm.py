# evaluation/evaluate_hybrid_svm.py
# ============================================================
# Evaluates the SVM-based Hybrid Model
# Plugs into your existing evaluate_models.py workflow
# ============================================================

import os
import sys
import numpy as np
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from data.dataset_loader import MITIndoorDataset


def evaluate_hybrid_svm():
    model_path = "models/hybrid_svm.pkl"
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        print("  Run: python training/train_hybrid_svm.py")
        return

    # Load class names from dataset so report is readable
    train_dataset = MITIndoorDataset(root_dir="data/MIT_Indoor/train", transform=None)
    class_names   = train_dataset.classes   # sorted folder names
    valid_labels  = list(range(len(class_names)))

    # Load test features
    test_path = "data/hog_features_test.npz"
    data      = np.load(test_path)
    X_test    = data["features"].astype(np.float32)
    y_test    = data["labels"].astype(np.int64)

    # Load pipeline and predict
    pipeline   = joblib.load(model_path)
    test_preds = pipeline.predict(X_test)

    acc = accuracy_score(y_test, test_preds)
    cm  = confusion_matrix(y_test, test_preds, labels=valid_labels)

    print("\n================ HYBRID SVM MODEL =================")
    print(f"Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(
        y_test, test_preds,
        labels=valid_labels,
        target_names=class_names,
        zero_division=0
    ))


if __name__ == "__main__":
    evaluate_hybrid_svm()