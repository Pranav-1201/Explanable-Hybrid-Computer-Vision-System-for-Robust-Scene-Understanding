# preprocessing/build_hog_pca.py
"""
Fit PCA(256) on training HOG features and save compressed train/test sets.
Must be re-run if hog_features_train.npz is regenerated.
Saves:
  data/hog_pca_model.pkl        — fitted PCA transformer
  data/hog_pca_train.npz        — (5360, 256) compressed train features
  data/hog_pca_test.npz         — (1340, 256) compressed test features
"""
import numpy as np, joblib, os, time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

HOG_TRAIN = 'data/hog_features_train.npz'
HOG_TEST  = 'data/hog_features_test.npz'
PCA_DIMS  = 512

if __name__ == '__main__':
    print("Loading HOG features...")
    train_data = np.load(HOG_TRAIN)
    X_train    = train_data['features'].astype(np.float32)
    y_train    = train_data['labels']
    X_test     = np.load(HOG_TEST)['features'].astype(np.float32)
    y_test     = np.load(HOG_TEST)['labels']
    print(f"  Train: {X_train.shape}  Test: {X_test.shape}")

    # Scale first, then PCA
    print(f"\nFitting StandardScaler + PCA({PCA_DIMS})...")
    t0      = time.time()
    scaler  = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    pca = PCA(n_components=PCA_DIMS, whiten=True, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca  = pca.transform(X_test_scaled)
    print(f"  [DONE] {time.time()-t0:.1f}s")
    print(f"  Explained variance ratio (total): {pca.explained_variance_ratio_.sum()*100:.1f}%")

    # Save scaler + PCA as a single bundle for use in app.py
    bundle = {'scaler': scaler, 'pca': pca}
    joblib.dump(bundle, 'data/hog_pca_model.pkl')
    np.savez('data/hog_pca_train.npz', features=X_train_pca, labels=y_train)
    np.savez('data/hog_pca_test.npz',  features=X_test_pca,  labels=y_test)
    print(f"\n[SAVED] PCA model   → data/hog_pca_model.pkl")
    print(f"[SAVED] Train PCA   → data/hog_pca_train.npz  {X_train_pca.shape}")
    print(f"[SAVED] Test PCA    → data/hog_pca_test.npz   {X_test_pca.shape}")
