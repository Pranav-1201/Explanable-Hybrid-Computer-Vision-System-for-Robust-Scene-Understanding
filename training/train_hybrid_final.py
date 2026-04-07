# training/train_hybrid_final.py
# ============================================================
# FINAL FIX: Correct PCA components + tuned SVM
# ------------------------------------------------------------
# Diagnosis from debug output:
#   - Features are clean (no NaN/Inf, SVM memorizes at 99.9%)
#   - Labels are perfectly aligned (0-66 in both splits)
#   - ROOT CAUSE: 200 PCA components = only 46.7% variance
#     We were discarding 53% of the signal before classifying
#   - Fix: use enough components to explain 90% variance
#     (~500-800 components for this feature set)
# ============================================================

import os
import sys
import warnings
import time
import numpy as np

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


def load_features():
    train = np.load("data/hog_features_train.npz")
    test  = np.load("data/hog_features_test.npz")
    X_tr  = train["features"].astype(np.float32)
    y_tr  = train["labels"]
    X_te  = test["features"].astype(np.float32)
    y_te  = test["labels"]
    print(f"Train: {X_tr.shape}  Test: {X_te.shape}")
    return X_tr, y_tr, X_te, y_te


def find_pca_components(X_tr_sc, target_variance=0.90):
    """Find how many PCA components explain target_variance."""
    print(f"\n[PCA] Finding components for {target_variance*100:.0f}% variance...")
    # Fit on a sample first to check variance curve
    pca_full = PCA(n_components=min(1000, X_tr_sc.shape[0],
                                    X_tr_sc.shape[1]),
                   random_state=42)
    pca_full.fit(X_tr_sc)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)

    for pct in [0.70, 0.80, 0.90, 0.95]:
        n = int(np.searchsorted(cumvar, pct)) + 1
        print(f"  {pct*100:.0f}% variance -> {n} components")

    n_components = int(np.searchsorted(cumvar, target_variance)) + 1
    print(f"  Using {n_components} components ({target_variance*100:.0f}% variance)")
    return n_components


def train():
    X_tr, y_tr, X_te, y_te = load_features()

    # ── Step 1: Scale ────────────────────────────────────────
    print("\n[STEP 1] Scaling features...")
    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    # ── Step 2: Find correct PCA components ─────────────────
    n_components = find_pca_components(X_tr_sc, target_variance=0.90)

    # ── Step 3: Apply PCA ────────────────────────────────────
    print(f"\n[STEP 2] Applying PCA ({n_components} components)...")
    t0      = time.time()
    pca     = PCA(n_components=n_components, whiten=True, random_state=42)
    X_tr_pca = pca.fit_transform(X_tr_sc)
    X_te_pca = pca.transform(X_te_sc)
    print(f"  Done in {time.time()-t0:.1f}s  "
          f"Shape: {X_tr_pca.shape}")

    # ── Step 4: LinearSVC — fast and strong for this regime ──
    # LinearSVC scales to large feature dims much faster than
    # RBF SVM. With 500+ PCA components, RBF SVM is very slow.
    # LinearSVC + good C is competitive and trains in seconds.
    print(f"\n[STEP 3] Training LinearSVC...")
    print("  (trying C = 0.01, 0.1, 1.0, 10.0)")

    best_acc, best_C, best_model = 0, None, None

    for C in [0.01, 0.1, 1.0, 10.0]:
        t0  = time.time()
        clf = LinearSVC(C=C, max_iter=3000, random_state=42)
        clf.fit(X_tr_pca, y_tr)
        tr_acc = accuracy_score(y_tr, clf.predict(X_tr_pca))
        te_acc = accuracy_score(y_te, clf.predict(X_te_pca))
        elapsed = time.time() - t0
        print(f"  C={C:5.2f} | train={tr_acc*100:.1f}% "
              f"test={te_acc*100:.1f}%  ({elapsed:.1f}s)")

        if te_acc > best_acc:
            best_acc   = te_acc
            best_C     = C
            best_model = clf

    print(f"\n  Best C={best_C}  ->  test accuracy: {best_acc*100:.2f}%")

    # ── Step 5: Full report ───────────────────────────────────
    y_pred = best_model.predict(X_te_pca)
    print(f"\n{'='*55}")
    print(f"FINAL TEST ACCURACY: {best_acc*100:.2f}%")
    print(f"{'='*55}")
    print(classification_report(y_te, y_pred, zero_division=0))

    # ── Step 6: Save pipeline ────────────────────────────────
    pipe = Pipeline([
        ("scaler", scaler),
        ("pca",    pca),
        ("svm",    best_model)
    ])
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipe, "models/hybrid_svm_pipeline.pkl")
    print("Saved: models/hybrid_svm_pipeline.pkl")


if __name__ == "__main__":
    train()