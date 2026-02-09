import cv2
import matplotlib.pyplot as plt
from texture import gabor_features

# Load sample image
img_path = "data/MIT_Indoor/train/kitchen"
img_file = next(iter(__import__('os').listdir(img_path)))

image = cv2.imread(f"{img_path}/{img_file}")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gabor_img = gabor_features(gray)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(gray, cmap="gray")
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Gabor Features")
plt.imshow(gabor_img, cmap="gray")
plt.axis("off")

plt.show()
