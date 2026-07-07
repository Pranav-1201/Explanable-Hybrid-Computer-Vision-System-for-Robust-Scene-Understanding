# preprocessing/build_fusion_dataset.py
"""
Merge CNN embeddings and PCA-HOG features into a single dataset file.
Requires: data/cnn_embeddings_{train,test}.npz
          data/hog_pca_{train,test}.npz
Produces: data/fusion_train.npz — {'cnn': (5360,2048), 'hog': (5360,256), 'labels': (5360,)}
          data/fusion_test.npz  — {'cnn': (1340,2048), 'hog': (1340,256), 'labels': (1340,)}
"""
import numpy as np

def build(split):
    cnn_data  = np.load(f'data/cnn_embeddings_{split}.npz')
    hog_data  = np.load(f'data/hog_pca_{split}.npz')
    assert np.array_equal(cnn_data['labels'], hog_data['labels']), \
        f"Label mismatch between CNN and HOG {split} sets!"
    np.savez(
        f'data/fusion_{split}.npz',
        cnn    = cnn_data['embeddings'],
        hog    = hog_data['features'],
        labels = cnn_data['labels'],
    )
    print(f"[{split.upper()}] CNN: {cnn_data['embeddings'].shape} | "
          f"HOG: {hog_data['features'].shape} | "
          f"Labels: {cnn_data['labels'].shape}")

if __name__ == '__main__':
    build('train')
    build('test')
    print("\n[DONE] Fusion dataset files ready.")
    print("  NOTE: Run this script again after extracting new CNN embeddings.")
