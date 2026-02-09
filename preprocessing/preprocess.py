import cv2
import numpy as np


def to_grayscale(image):
    """Convert RGB image to grayscale"""
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def gaussian_blur(image, ksize=5):
    """Apply Gaussian blur"""
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def median_blur(image, ksize=5):
    """Apply Median blur"""
    return cv2.medianBlur(image, ksize)


def histogram_equalization(image):
    """
    Apply CLAHE for better local contrast enhancement
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(image)



def preprocess_image(image):
    """
    Full preprocessing pipeline
    Input: RGB image (numpy array)
    Output: preprocessed image
    """
    gray = to_grayscale(image)
    blurred = gaussian_blur(gray)
    enhanced = histogram_equalization(blurred)
    return enhanced


