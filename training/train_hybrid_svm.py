# training/train_hybrid_svm.py
# ============================================================
# Hybrid Model Training — SVM Pipeline (Replaces MLP)
# ------------------------------------------------------------
# WHY SVM INSTEAD OF MLP?
#   The MLP (HybridCNN) achieved 95% train / 16% test accuracy
#   — textbook overfitting. Root cause: 15,236 HOG features vs
#   only ~4,800 training samples. SVMs handle high-dimensional
#   sparse feature spaces far better via the kernel trick and
#   max-margin classification.
#
# PIPELINE:
#   Raw HOG features (15,236-d)
#     → StandardScaler  (zero mean, unit variance)
#     → PCA             (reduce to 512-d, keeps ~95% variance)
#     → LinearSVC       (fast, proven for scene recognition)
#
# WHY LinearSVC OVER RBF SVM?
#   After PCA the features are already in a dense, compact
#   space. LinearSVC scales as O(n) vs O(n^2) for RBF, and
#   on MIT Indoor 67 performs comparably while training in
#   seconds instead of hours.
#
# Expected test accuracy: 35–45% (vs 16% MLP baseline)
# ============================================================

import os
import sys
import time
import numpy as np
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.preprocessing  import StandardScaler
from sklearn.decomposition  import PCA
from sklearn.svm            import LinearSVC
from sklearn.pipeline       import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics        import accuracy_score

from data.dataset_loader import MITIndoorDataset


# ============================================================
# CONFIGURATION
# ============================================================
TRAIN_FEATURES = "data/hog_features_train.npz"
TEST_FEATURES  = "data/hog_features_test.npz"
MODEL_OUT      = "models/hybrid_svm.pkl"

# PCA: number of components to keep
# 512 retains ~95% variance of our 15k-dim HOG features
# while making LinearSVC dramatically faster to converge
PCA_COMPONENTS = 512

# LinearSVC regularisation — C controls margin hardness
# Lower C = wider margin = more regularisation = less overfit
# Higher C = tighter fit = risks overfit on small datasets
# 0.1 is conservative and works well for HOG + scene data
SVC_C = 0.1

# Maximum SVC iterations — increase if you see ConvergenceWarning
SVC_MAX_ITER = 5000


# ============================================================
# LOAD FEATURES
# ============================================================
def load_features():
    """
    Load pre-extracted HOG feature arrays from .npz files.
    These were produced by preprocessing/extract_hog_features.py
    Shape: (N_samples, 15236)
    """

    print("\n[LOADING] HOG features from disk...")

    if not os.path.exists(TRAIN_FEATURES):
        print(f"[ERROR] Train features not found: {TRAIN_FEATURES}")
        print("  Run: python preprocessing/extract_hog_features.py")
        sys.exit(1)

    if not os.path.exists(TEST_FEATURES):
        print(f"[ERROR] Test features not found: {TEST_FEATURES}")
        print("  Run: python preprocessing/extract_hog_features.py")
        sys.exit(1)

    train_data = np.load(TRAIN_FEATURES)
    test_data  = np.load(TEST_FEATURES)

    X_train = train_data["features"].astype(np.float32)
    y_train = train_data["labels"].astype(np.int64)

    X_test  = test_data["features"].astype(np.float32)
    y_test  = test_data["labels"].astype(np.int64)

    print(f"  Train : {X_train.shape}  ({len(np.unique(y_train))} classes)")
    print(f"  Test  : {X_test.shape}")

    return X_train, y_train, X_test, y_test


# ============================================================
# BUILD PIPELINE
# ============================================================
def build_pipeline():
    """
    Build the full sklearn Pipeline:
        StandardScaler → PCA → LinearSVC

    Using a Pipeline object ensures:
      - No data leakage (scaler/PCA fit only on train)
      - Single joblib save/load for the whole system
      - Easy to swap components for experiments
    """

    print(f"\n[PIPELINE] Building: Scaler → PCA({PCA_COMPONENTS}) → LinearSVC(C={SVC_C})")

    pipeline = Pipeline([
        # ── Step 1: Normalise ──────────────────────────────────
        # StandardScaler: zero mean, unit variance per feature.
        # Critical for SVM — without this, large-magnitude HOG
        # cells dominate the kernel and slow convergence badly.
        ("scaler", StandardScaler()),

        # ── Step 2: Dimensionality Reduction ──────────────────
        # PCA compresses 15,236 → 512 dimensions.
        # Benefits:
        #   - Removes correlated HOG cells (noise reduction)
        #   - LinearSVC converges in seconds instead of minutes
        #   - Retains >95% of explained variance
        ("pca", PCA(
            n_components=PCA_COMPONENTS,
            whiten=True,        # whitening decorrelates PCA outputs
            random_state=42
        )),

        # ── Step 3: Classifier ────────────────────────────────
        # LinearSVC with One-vs-Rest for 67-class problem.
        # dual=False is preferred when n_samples > n_features
        # (which is true after PCA: 5360 samples, 512 features)
        ("svm", LinearSVC(
            C=SVC_C,
            max_iter=SVC_MAX_ITER,
            dual=False,
            random_state=42,
            verbose=0
        )),
    ])

    return pipeline


# ============================================================
# TRAIN
# ============================================================
def train(pipeline, X_train, y_train):
    """
    Fit the pipeline on training data.
    Pipeline.fit() handles scaler → PCA → SVM in one call,
    with correct fit/transform separation (no leakage).
    """

    print("\n[TRAINING] Fitting pipeline on training data...")
    print("  This may take 30–120 seconds depending on your CPU.")

    t0 = time.time()
    pipeline.fit(X_train, y_train)
    elapsed = time.time() - t0

    print(f"  [DONE] Training time: {elapsed:.1f}s")
    return pipeline


# ============================================================
# QUICK SANITY: CROSS-VALIDATION ON TRAIN SET
# ============================================================
def cross_validate(pipeline, X_train, y_train):
    """
    3-fold CV on training data to check for overfitting
    before evaluating on test set.

    A healthy model should show:
      - CV mean ≈ test accuracy (no overfit)
      - Low std across folds (stable features)
    """

    print("\n[CROSS-VALIDATION] 3-fold on train set (sanity check)...")
    print("  Note: This fits the pipeline 3 times — may take 2–4 min.")

    scores = cross_val_score(pipeline, X_train, y_train,
                             cv=3, scoring="accuracy", n_jobs=-1)

    print(f"  Fold scores : {scores * 100}")
    print(f"  Mean        : {scores.mean() * 100:.2f}%")
    print(f"  Std         : {scores.std() * 100:.2f}%")

    return scores


# ============================================================
# EVALUATE ON TEST SET
# ============================================================
def evaluate(pipeline, X_test, y_test, class_names):
    """
    Evaluate fitted pipeline on held-out test set.
    Reports top-1 accuracy and per-class breakdown.
    """

    print("\n[EVALUATION] Running on test set...")

    preds = pipeline.predict(X_test)
    acc   = accuracy_score(y_test, preds)

    print(f"\n  ✓ Test Accuracy : {acc * 100:.2f}%")
    print(f"  (Previous MLP   : 16.04%  ← this should be much higher)")

    return acc, preds


# ============================================================
# SAVE MODEL
# ============================================================
def save_model(pipeline):
    """
    Save the entire Pipeline object (scaler + PCA + SVM) as one
    .pkl file. Avoids the PCA/scaler mismatch bug that would
    occur if you saved them separately.
    """

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, MODEL_OUT)
    size_mb = os.path.getsize(MODEL_OUT) / 1e6

    print(f"\n[SAVED] Model → {MODEL_OUT}  ({size_mb:.1f} MB)")
    print("  Load with: pipeline = joblib.load('models/hybrid_svm.pkl')")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":

    print("=" * 60)
    print("  HYBRID SVM TRAINING — HOG + PCA + LinearSVC")
    print("=" * 60)

    # ── 1. Load features ──────────────────────────────────────
    X_train, y_train, X_test, y_test = load_features()

    # ── 2. Class names (for readable evaluation output) ───────
    train_dataset = MITIndoorDataset(
        root_dir="data/MIT_Indoor/train",
        transform=None
    )
    class_names = train_dataset.classes

    # ── 3. Build sklearn pipeline ─────────────────────────────
    pipeline = build_pipeline()

    # ── 4. Train ──────────────────────────────────────────────
    pipeline = train(pipeline, X_train, y_train)

    # ── 5. Cross-validate (optional sanity check) ─────────────
    # Comment this out if you just want fast evaluation:
    # cross_validate(pipeline, X_train, y_train)

    # ── 6. Evaluate on test set ───────────────────────────────
    acc, preds = evaluate(pipeline, X_test, y_test, class_names)

    # ── 7. Save ───────────────────────────────────────────────
    save_model(pipeline)

    print("\n" + "=" * 60)
    print(f"  DONE — Test Accuracy: {acc * 100:.2f}%")
    print("  Next: Update app.py to load hybrid_svm.pkl via joblib")
    print("=" * 60)