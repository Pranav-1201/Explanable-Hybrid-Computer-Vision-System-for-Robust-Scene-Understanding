import cv2
import matplotlib.pyplot as plt
from preprocess import preprocess_image

# Load a sample image (change path if needed)
img_path = "data/MIT_Indoor/train/kitchen"
img_file = next(iter(__import__('os').listdir(img_path)))

image = cv2.imread(f"{img_path}/{img_file}")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

processed = preprocess_image(image)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(image)
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Preprocessed")
plt.imshow(processed, cmap="gray")
plt.axis("off")

plt.show()
