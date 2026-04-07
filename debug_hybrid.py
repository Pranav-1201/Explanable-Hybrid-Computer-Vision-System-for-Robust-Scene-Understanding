import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

print("=" * 60)
print("HYBRID MODEL DIAGNOSTIC")
print("=" * 60)

train = np.load("data/hog_features_train.npz")
test  = np.load("data/hog_features_test.npz")
X_tr, y_tr = train["features"].astype(np.float32), train["labels"]
X_te, y_te = test["features"].astype(np.float32),  test["labels"]

print(f"\n[DATA]")
print(f"  Train shape       : {X_tr.shape}")
print(f"  Test  shape       : {X_te.shape}")
print(f"  Train label range : {y_tr.min()} - {y_tr.max()}")
print(f"  Test  label range : {y_te.min()} - {y_te.max()}")
print(f"  Train classes     : {len(np.unique(y_tr))}")
print(f"  Test  classes     : {len(np.unique(y_te))}")

in_train_not_test = set(np.unique(y_tr)) - set(np.unique(y_te))
in_test_not_train = set(np.unique(y_te)) - set(np.unique(y_tr))
print(f"  In train NOT test : {in_train_not_test}")
print(f"  In test  NOT train: {in_test_not_train}")

print(f"\n[FEATURE SANITY]")
print(f"  NaN count     : {np.isnan(X_tr).sum()}")
print(f"  Inf count     : {np.isinf(X_tr).sum()}")
print(f"  All-zero rows : {(X_tr == 0).all(axis=1).sum()}")
print(f"  Mean / std    : {X_tr.mean():.4f} / {X_tr.std():.4f}")

# Fit scaler+PCA on train only
scaler   = StandardScaler()
X_tr_sc  = scaler.fit_transform(X_tr)
X_te_sc  = scaler.transform(X_te)

pca      = PCA(n_components=200, random_state=42)
X_tr_pca = pca.fit_transform(X_tr_sc)
X_te_pca = pca.transform(X_te_sc)

var_exp = pca.explained_variance_ratio_.sum()
print(f"\n[PCA] 200 components explain {var_exp*100:.1f}% variance")

# Logistic Regression — fast, good baseline
print(f"\n[LOGISTIC REGRESSION]")
lr = LogisticRegression(C=1.0, max_iter=1000,
                        solver="saga", n_jobs=-1, random_state=42)
lr.fit(X_tr_pca, y_tr)
lr_train = accuracy_score(y_tr, lr.predict(X_tr_pca))
lr_test  = accuracy_score(y_te, lr.predict(X_te_pca))
print(f"  Train acc : {lr_train*100:.2f}%")
print(f"  Test  acc : {lr_test*100:.2f}%")
print(f"  Overfit gap: {(lr_train-lr_test)*100:.2f}%")

# Per-class breakdown
print(f"\n[PER-CLASS TEST ACCURACY]")
y_pred = lr.predict(X_te_pca)
for cls in range(int(y_te.max()) + 1):
    mask = y_te == cls
    if mask.sum() == 0:
        continue
    acc = accuracy_score(y_te[mask], y_pred[mask])
    print(f"  Class {cls:2d}: {acc*100:5.1f}%  (n={mask.sum()})")

# Random baseline
rng = np.random.default_rng(42)
rand_acc = accuracy_score(y_te, rng.choice(np.unique(y_te), size=len(y_te)))
print(f"\n[RANDOM BASELINE]: {rand_acc*100:.2f}%  (expect ~1.5%)")
print("=" * 60)