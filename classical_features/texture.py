import cv2
import numpy as np
from skimage.feature import hog


import cv2
from skimage.feature import hog

def hog_features(image):
    """
    Extract HOG features from a PIL or NumPy image.
    """

    # --- Convert PIL Image to NumPy array ---
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    # --- Convert RGB to grayscale if needed ---
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    image = cv2.resize(image, (128, 128))

    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False,
        block_norm='L2-Hys'
    )

    return features


def gabor_features(image):
    """
    Multi-orientation Gabor filters for richer texture extraction
    """
    responses = []
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        kernel = cv2.getGaborKernel(
            ksize=(31, 31),
            sigma=4.0,
            theta=theta,
            lambd=10.0,
            gamma=0.5,
            psi=0
        )
        filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
        responses.append(filtered)

    return np.mean(responses, axis=0)

