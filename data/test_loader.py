import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataset_loader import MITIndoorDataset, get_transforms
from torch.utils.data import DataLoader

train_dataset = MITIndoorDataset(
    root_dir="data/MIT_Indoor/train",
    transform=get_transforms()
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

images, labels = next(iter(train_loader))

print("Image batch shape:", images.shape)
print("Labels:", labels)
