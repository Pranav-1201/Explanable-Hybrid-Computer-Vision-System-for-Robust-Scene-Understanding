import cv2
import matplotlib.pyplot as plt
from edges import canny_edges, log_edges

# Load sample image
img_path = "data/MIT_Indoor/train/kitchen"
img_file = next(iter(__import__('os').listdir(img_path)))

image = cv2.imread(f"{img_path}/{img_file}")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

canny = canny_edges(gray)
log = log_edges(gray)

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Original")
plt.imshow(gray, cmap="gray")
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Canny Edges")
plt.imshow(canny, cmap="gray")
plt.axis("off")

plt.subplot(1,3,3)
plt.title("LoG Edges")
plt.imshow(log, cmap="gray")
plt.axis("off")

plt.show()
