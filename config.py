# config.py

import os

# Dataset and Project Configuration
PARENT_DIR = ''  # Please update to your project's root directory
SAVE_FOLDER_NAME = "CFZ_Segementation_TransUnet_V01T01"
DATAFRAME_NAME = 'train_data_CFZ.csv'
IMAGESIZE = 320  # Set the standard image size for resizing images

# Derived Paths (do not modify directly)
SAVE_PATH = os.path.join(PARENT_DIR, SAVE_FOLDER_NAME)
PROJECT_PATH = os.path.join(PARENT_DIR, 'tmp_CFZ')
DATASET_PATH = os.path.join(PROJECT_PATH, 'Dataset')
TRAIN_PATH = os.path.join(DATASET_PATH, 'train')
INPUT_PATH = os.path.join(TRAIN_PATH, 'Input')
OUTPUT_PATH = os.path.join(TRAIN_PATH, 'CFZ_map')

# Model and Training Parameters
CLASSES = ['background', 'vCFZ', 'vein', 'aCFZ', 'artery']
NUM_OF_CLASSES = len(CLASSES)
CLASS_COLORS = [(0, 0, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (255, 0, 0)]
CLASS_WEIGHTS = [1, 1, 1, 1, 1]  # Update as needed
ID2LABEL = {k: v for k, v in enumerate(CLASSES)}
LABEL2ID = {v: k for k, v in enumerate(CLASSES)}
CLASSES_TO_TRAIN = CLASSES  # Specify which classes to include in training

BATCH_SIZE = 15
FOLD_NUM = 5  # Current fold for cross-validation
MAX_FOLD = 5  # Total number of folds

# Data Source (if using the data transfer function)
SOURCE_DIR = "" # Please update to your raw data source directory
