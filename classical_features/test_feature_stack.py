import cv2
from classical_features.feature_stack import stack_features


# Load sample image
img_path = "data/MIT_Indoor/train/kitchen"
img_file = next(iter(__import__('os').listdir(img_path)))

image = cv2.imread(f"{img_path}/{img_file}")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

stacked = stack_features(image)

print("Stacked tensor shape:", stacked.shape)
