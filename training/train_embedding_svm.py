# training/train_embedding_svm.py
"""
Architecture A: CNN Embedding (2048-d) → PCA(512) → LinearSVC
This is the drop-in replacement for the broken HOG→SVM pipeline.
Expected accuracy: 45–60% (vs 10.75% for HOG→SVM)
"""
import numpy as np, joblib, os, time
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report
from data.dataset_loader import MITIndoorDataset

def load_embeddings(split):
    path = f'data/cnn_embeddings_{split}.npz'
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run: python preprocessing/extract_cnn_embeddings.py"
        )
    data = np.load(path)
    return data['embeddings'], data['labels']

if __name__ == '__main__':
    print("=" * 60)
    print("  ARCHITECTURE A: CNN-EMBEDDING SVM")
    print("=" * 60)

    # Load embeddings
    print("\n[LOADING] CNN embeddings...")
    X_train, y_train = load_embeddings('train')
    X_test,  y_test  = load_embeddings('test')
    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"  Classes: {len(np.unique(y_train))}")

    # Build pipeline: StandardScaler → PCA(512) → CalibratedLinearSVC
    # CalibratedClassifierCV wraps LinearSVC to provide predict_proba()
    svm = LinearSVC(C=0.1, max_iter=5000, dual=False)
    calibrated_svm = CalibratedClassifierCV(svm, cv=5, method='sigmoid')

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca',    PCA(n_components=512, whiten=True, random_state=42)),
        ('svm',    calibrated_svm),
    ])

    print("\n[TRAINING] CNN-Embedding SVM pipeline...")
    print("  Scaler → PCA(512, whitened) → CalibratedLinearSVC(C=0.1)")
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    print(f"  [DONE] Training time: {time.time()-t0:.1f}s")

    # Evaluate
    y_pred = pipeline.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\n  ✓ Test Accuracy : {acc*100:.2f}%")
    print(f"  (HOG SVM baseline: 10.75%  ← this should be dramatically higher)")

    # Load class names
    ds = MITIndoorDataset('data/MIT_Indoor/test', transform=None)
    class_names = [ds.classes[i] for i in range(len(ds.classes))]
    print("\n" + classification_report(y_test, y_pred, target_names=class_names))

    # Save
    os.makedirs('models', exist_ok=True)
    save_path = 'models/embedding_svm.pkl'
    joblib.dump(pipeline, save_path)
    print(f"\n[SAVED] {save_path}  ({os.path.getsize(save_path)/1e6:.1f} MB)")
    print("  Load: pipeline = joblib.load('models/embedding_svm.pkl')")
