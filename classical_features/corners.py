import cv2
import numpy as np


def harris_corners(image, block_size=2, ksize=3, k=0.04):
    """
    Harris corner detection
    Input: Grayscale image
    Output: Corner response map
    """
    image = np.float32(image)
    dst = cv2.cornerHarris(image, block_size, ksize, k)
    dst = cv2.dilate(dst, None)  # enhance corner points
    return dst


def mark_corners(image, corner_response, threshold=0.02):
    """
    Mark detected corners on the image
    """
    marked = image.copy()
    marked[corner_response > threshold * corner_response.max()] = 255
    return marked
