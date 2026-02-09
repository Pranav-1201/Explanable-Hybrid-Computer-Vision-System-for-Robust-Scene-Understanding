import cv2
import matplotlib.pyplot as plt
from corners import harris_corners, mark_corners

# Load sample image
img_path = "data/MIT_Indoor/train/kitchen"
img_file = next(iter(__import__('os').listdir(img_path)))

image = cv2.imread(f"{img_path}/{img_file}")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

corner_response = harris_corners(gray)
marked = mark_corners(gray, corner_response)

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(gray, cmap="gray")
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Harris Corners")
plt.imshow(marked, cmap="gray")
plt.axis("off")

plt.show()
