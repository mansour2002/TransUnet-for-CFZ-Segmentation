# TransUnet-for-CFZ-Segmentation

TransUnet for Capillary Free Zones (CFZ) Segmentation using PyTorch.

## Overview

This GitHub repository hosts the implementation of a TransUnet model developed in PyTorch for segmenting capillary free zones (CFZ), arteries, and veins in optical coherence tomography angiography (OCTA) images. The model is trained on high-resolution 6x6mm OCTA images, aiming to accurately delineate vascular structures for medical research and diagnosis.

**Key Features:**

* Utilizes PyTorch framework for efficient training and inference
* Designed specifically for segmenting CFZ, arteries, and veins in OCTA images
* Preprocessing scripts for data augmentation and preparation
* Evaluation metrics for assessing segmentation performance

Images were acquired using the AngioVue SD-OCT device (Optovue, Fremont, CA, USA). The OCT system had a 70,000 Hz A-scan rate with ~5 μm axial and ~15 μm lateral resolutions. All OCTA images used for this study were 6 mm × 6 mm scans; only superficial OCTA images were used.

These figures show two representative OCTA images and corresponding manually generated ground truths and predicted images:

![The CFZ-Net](https://github.com/mansour2002/TransUnet-for-CFZ-Segmentation/blob/main/Figures/Segmentation%201.png?raw=true)


![The CFZ-Net](https://github.com/mansour2002/TransUnet-for-CFZ-Segmentation/blob/main/Figures/Segmentation%202.png?raw=true)


## Repository Structure

The project is organized into the following key files for better modularity and professionalism:

* `config.py`: Contains all configurable parameters, paths, and hyperparameters.
* `model.py`: Defines the TransUnet model architecture.
* `data_utils.py`: Includes utilities for data loading, preprocessing, and augmentation, including the `SegmentationDataset` class.
* `data_preparation.py`: (Optional) A standalone script for initial data transfer and directory setup. Run this once to prepare your dataset.
* `train.py`: The main script to execute the training and validation process.

## Setup and Usage

### Dependencies

* PyTorch >= 2.2.1+cu118
* CUDA >= 11.8
* Python >= 3.9
* `segmentation_models_pytorch`
* `albumentations`
* `pandas`
* `opencv-python`
* `timm`
* `einops`
* `ml_collections` (if used by your specific TransUnet implementation)
* `torchmetrics`

You can install the required packages using pip:
```bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install segmentation_models_pytorch albumentations pandas opencv-python timm einops ml_collections torchmetrics
```

### Data Preparation

1.  **Update `config.py`**:
    Before running any scripts, open `config.py` and update `PARENT_DIR` and `SOURCE_DIR` to your local project and raw data paths, respectively.

2.  **Run Data Preparation (Optional, if your data is not already structured)**:
    Execute `data_preparation.py` to organize your raw OCTA images and CFZ maps into the required input/output structure and generate the `train_data_tmp_Idea61CFZ.csv` file.
    ```bash
    python data_preparation.py
    ```

### Training

1.  **Start Training**:
    Once your data is prepared and paths are correctly configured, you can start the training process by running `train.py`:
    ```bash
    python train.py
    ```
