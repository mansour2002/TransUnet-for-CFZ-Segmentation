# data_utils.py

import os
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
import albumentations as A
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# Import configurations
from config import (
    CLASSES, CLASS_COLORS, IMAGESIZE, DATAFRAME_NAME, PARENT_DIR,
    INPUT_PATH, OUTPUT_PATH
)

Tensor2PILImage = T.ToPILImage()
PILImage2Tensor = T.ToTensor()

def set_class_values(all_classes, classes_to_train):
    """Assigns a specific class label to each of the classes."""
    class_indices = {cls.lower(): idx for idx, cls in enumerate(all_classes)}
    class_values = [class_indices[cls.lower()] for cls in classes_to_train]
    return class_values

def get_label_mask(mask, class_values, label_colors_list):
    """Encodes image pixels into class-specific labels."""
    mask = np.array(mask)
    label_colors = np.array(label_colors_list)
    label_mask = np.zeros((mask.shape[0], mask.shape[1], 1), dtype=int)

    for value, color in enumerate(label_colors):
        if value in class_values:
            matches = np.all(mask == color, axis=-1)
            label_mask[matches] = value
    return label_mask

def load_image_paths(df, path, column):
    """Lists image paths."""
    images = []
    for i, item in df.iterrows():
        temp_item = os.path.join(path, item[column])
        images.append(temp_item)
    return images

def get_images(parent_dir, fold_num, max_fold, dataframe_name=DATAFRAME_NAME, input_path=INPUT_PATH, output_path=OUTPUT_PATH):
    """Loads paths for training and validation images and masks based on the fold."""
    train_csv_path = os.path.join(parent_dir, dataframe_name)
    df = pd.read_csv(train_csv_path)

    num_samples = len(df)
    start_val = int((fold_num - 1) / max_fold * num_samples)
    end_val = int(fold_num / max_fold * num_samples)

    df_train = pd.concat([df[:start_val], df[end_val:]], ignore_index=True)
    df_val = df[start_val:end_val]

    train_images = load_image_paths(df_train, input_path, 0)
    train_masks = load_image_paths(df_train, output_path, 1)
    valid_images = load_image_paths(df_val, input_path, 0)
    valid_masks = load_image_paths(df_val, output_path, 1)

    return train_images, train_masks, valid_images, valid_masks

def train_transforms(img_size):
    """Transforms/augmentations for training images and masks."""
    train_image_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(shear=0.4, mode=cv2.BORDER_REFLECT_101, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.2, p=1.0),
        A.ShiftScaleRotate(scale_limit=0.3, rotate_limit=50, shift_limit=0.3, p=1.0, border_mode=cv2.BORDER_REFLECT_101),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, always_apply=True, border_mode=cv2.BORDER_REFLECT_101),
        A.Blur(blur_limit=3, p=0.2),
        A.Resize(img_size, img_size, always_apply=True, p=1.0)
    ])
    return train_image_transform

def valid_transforms(img_size):
    """Transforms/augmentations for validation images and masks."""
    return A.Compose([A.Resize(img_size, img_size, always_apply=True, p=1.0)])

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, tfms, label_colors_list, classes_to_train, all_classes):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.tfms = tfms
        self.label_colors_list = label_colors_list
        self.all_classes = all_classes
        self.classes_to_train = classes_to_train
        self.class_values = set_class_values(self.all_classes, self.classes_to_train)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype('float32')
        image = image / 255.0
        mask = cv2.imread(self.mask_paths[index], cv2.IMREAD_COLOR)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB).astype('float32')

        transformed = self.tfms(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']

        mask = get_label_mask(mask, self.class_values, self.label_colors_list)

        image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
        mask = torch.from_numpy(np.transpose(mask, (2, 0, 1))).long()

        return image, mask