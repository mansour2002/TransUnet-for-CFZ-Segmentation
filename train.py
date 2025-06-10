# train.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as TM
from torch.utils.data import DataLoader
import os
import time

# Import configurations
from config import (
    PARENT_DIR, IMAGESIZE, CLASSES_TO_TRAIN, BATCH_SIZE,
    FOLD_NUM, MAX_FOLD, NUM_OF_CLASSES, CLASS_COLORS, CLASS_WEIGHTS
)

# Import data utilities and model
from data_utils import get_images, train_transforms, valid_transforms, SegmentationDataset
from model import TransUnet 

def get_device():
    """Return the appropriate device (CUDA or CPU) based on availability."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_device(x, device):
    """Send the tensor or model to the appropriate device."""
    return x.to(device)

def get_model_parameters(model):
    return sum(param.numel() for param in model.parameters())

def main():
    device = get_device()
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Load images and masks
    print("Loading image paths...")
    train_images, train_masks, valid_images, valid_masks = get_images(
        parent_dir=PARENT_DIR, fold_num=FOLD_NUM, max_fold=MAX_FOLD
    )
    print(f"Found {len(train_images)} training images and {len(valid_images)} validation images.")

    # Set up transformations
    train_tfms = train_transforms(IMAGESIZE)
    valid_tfms = valid_transforms(IMAGESIZE)

    # Initialize datasets
    train_dataset = SegmentationDataset(
        train_images, train_masks, train_tfms,
        CLASS_COLORS, CLASSES_TO_TRAIN, CLASSES
    )
    valid_dataset = SegmentationDataset(
        valid_images, valid_masks, valid_tfms,
        CLASS_COLORS, CLASSES_TO_TRAIN, CLASSES
    )

    # Configure data loaders
    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    valid_data_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    print(f"Train DataLoader batches: {len(train_data_loader)}")
    print(f"Valid DataLoader batches: {len(valid_data_loader)}")

   
    # Example: If your TransUnet uses a specific encoder.
    model = TransUnet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=NUM_OF_CLASSES)
    model = to_device(model, device)
    print(f"Model parameters: {get_model_parameters(model)}")

    # Define loss function and optimizer
    # Use class_weights if defined in config.py
    loss_weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=loss_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # Example optimizer

    # Training loop (simplified for brevity)
    num_epochs = 10 # Example number of epochs
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        start_time = time.time()
        for batch_idx, (inputs, targets) in enumerate(train_data_loader):
            inputs, targets = to_device(inputs, device), to_device(targets, device)
            targets = targets.squeeze(1) # Remove channel dimension for CrossEntropyLoss

            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Ensure outputs and targets have compatible shapes
            # outputs: (batch_size, num_classes, H, W)
            # targets: (batch_size, H, W) where values are class indices
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_data_loader)
        end_time = time.time()
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Time: {end_time - start_time:.2f}s")

        # Validation phase (simplified)
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for inputs, targets in valid_data_loader:
                inputs, targets = to_device(inputs, device), to_device(targets, device)
                targets = targets.squeeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                valid_loss += loss.item()
        avg_valid_loss = valid_loss / len(valid_data_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Valid Loss: {avg_valid_loss:.4f}")

    print("Training complete.")
    # You might want to save the model here
    torch.save(model.state_dict(), os.path.join(SAVE_PATH, "transunet_model.pth"))

if __name__ == '__main__':
    # Ensure data is prepared before training
    # You would typically run data_preparation.py once manually or have it as a separate step
    # For demonstration, we assume data directories and CSV are already set up.
    main()