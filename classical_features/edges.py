import cv2
import numpy as np


def canny_edges(image):
    """
    Adaptive Canny edge detection
    Apply Canny edge detection
    Input: Grayscale image
    Output: Edge map
    """
    v = np.median(image)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    edges = cv2.Canny(image, lower, upper)
    return edges


def log_edges(image):
    """
    Laplacian of Gaussian edge detection
    """
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    log = cv2.Laplacian(blurred, cv2.CV_64F)
    log = np.uint8(np.absolute(log))
    return log
