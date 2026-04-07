"""
Transfer Learning Model Training
-----------------------------------------

This script trains a pretrained ResNet50 model on the MIT Indoor dataset
using transfer learning.

Main Steps:
1. Load dataset using the existing project dataset loader
2. Load pretrained ResNet50
3. Replace final classification layer
4. Train the model
5. Save trained weights

Output:
models/transfer_resnet50.pth
"""

# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import time

# Import existing dataset loader from project
from data.dataset_loader import get_train_loader


def main():
        
    # ==========================================
    # 2. DEVICE CONFIGURATION
    # ==========================================

    # Use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Enable fastest GPU algorithms
    torch.backends.cudnn.benchmark = True

    print("Using device:", device)


    # ==========================================
    # 3. LOAD TRAINING DATA
    # ==========================================

    # Larger batch size for RTX 4060
    train_loader = get_train_loader(batch_size=64)

    # Number of classes in dataset
    num_classes = len(train_loader.dataset.classes)

    print("Number of classes:", num_classes)


    # ==========================================
    # 4. LOAD PRETRAINED RESNET50 MODEL
    # ==========================================

    print("Loading pretrained ResNet50...")

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Freeze pretrained layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final fully connected layer
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    )

    # Fine-tune last two ResNet stages
    for param in model.layer3.parameters():
        param.requires_grad = True

    for param in model.layer4.parameters():
        param.requires_grad = True

    # Move model to GPU / CPU
    model = model.to(device)

    print("Model ready.")


    # ==========================================
    # 5. DEFINE LOSS FUNCTION & OPTIMIZER
    # ==========================================

    # Cross entropy for multi-class classification
    criterion = nn.CrossEntropyLoss()

    # Train only trainable parameters
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=1e-4
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,
        gamma=0.3
    )

    # Mixed precision scaler (RTX acceleration)
    scaler = torch.cuda.amp.GradScaler()

    total_time = 0

    # ==========================================
    # 6. TRAINING LOOP
    # ==========================================

    import time

    epochs = 30

    print("Starting training...")

    total_time = 0

    for epoch in range(epochs):

        epoch_start = time.time()

        model.train()

        running_loss = 0

        for images, labels in train_loader:

            # Move data to device
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Reset gradients
            optimizer.zero_grad()

            # Mixed precision forward pass
            with torch.cuda.amp.autocast():

                outputs = model(images)

                # Compute loss
                loss = criterion(outputs, labels)

            # Backpropagation
            scaler.scale(loss).backward()

            # Update weights
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        scheduler.step()

        avg_loss = running_loss / len(train_loader)

        epoch_time = time.time() - epoch_start
        total_time += epoch_time

        avg_epoch_time = total_time / (epoch + 1)

        remaining_epochs = epochs - (epoch + 1)
        eta = remaining_epochs * avg_epoch_time

        print(
            f"Epoch [{epoch+1}/{epochs}]  "
            f"Loss: {avg_loss:.4f}  "
            f"Time: {epoch_time:.2f}s  "
            f"Avg: {avg_epoch_time:.2f}s  "
            f"ETA: {eta/60:.2f} min"
        )

    final_avg = total_time / epochs

    print("\n================ TRAINING SUMMARY =================")
    print(f"Total Training Time: {total_time/60:.2f} minutes")
    print(f"Average Epoch Time: {final_avg:.2f} seconds")


    # ==========================================
    # 7. SAVE TRAINED MODEL
    # ==========================================

    torch.save(model.state_dict(), "models/transfer_resnet50.pth")

    print("Training complete.")
    print("Model saved at: models/transfer_resnet50.pth")


if __name__ == "__main__":
    main()