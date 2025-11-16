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


## Model Architecture

This implementation uses **TransUnet (R50-ViT-B/16)**, which combines:
- **ResNet50** as the CNN backbone for feature extraction
- **Vision Transformer (ViT-B/16)** encoder with 12 transformer layers
- **U-Net style decoder** with skip connections for precise segmentation

The model segments OCTA images into 5 classes:
1. Background
2. Venous Capillary Free Zone (vCFZ)
3. Vein
4. Arterial Capillary Free Zone (aCFZ)
5. Artery

## Repository Structure

The project is organized into the following key files for better modularity and professionalism:

* [config.py](config.py): Contains all configurable parameters, paths, and hyperparameters
* [model.py](model.py): Defines the TransUnet model architecture (R50-ViT-B/16)
* [data_utils.py](data_utils.py): Utilities for data loading, preprocessing, and augmentation
* [data_preparation.py](data_preparation.py): Standalone script for dataset preparation and preprocessing
* [train.py](train.py): Main training script with validation loop
* [notebooks/](notebooks/): Contains the original training notebook with complete pipeline

## Setup and Usage

### Dependencies

* Python >= 3.8
* PyTorch >= 2.0.0
* CUDA (optional, for GPU acceleration)

You can install all required packages using pip:
```bash
pip install -r requirements.txt
```

For GPU support with CUDA, install PyTorch with CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Data Preparation

1.  **Update Configuration**:
    Before running any scripts, open [config.py](config.py) and update the following paths:
    - `PARENT_DIR`: Your project's root directory
    - `SOURCE_DIR`: Path to your raw OCTA data (if using data_preparation.py)

2.  **Prepare Your Dataset** (if not already structured):
    If you have raw OCTA images that need preprocessing, run:
    ```bash
    python data_preparation.py
    ```
    This script will:
    - Create necessary directory structure
    - Process and resize OCTA images and CFZ maps
    - Generate a CSV file with image paths for training

    Expected directory structure after preparation:
    ```
    PARENT_DIR/
    ├── tmp_CFZ/
    │   └── Dataset/
    │       └── train/
    │           ├── Input/      # Processed OCTA images
    │           └── CFZ_map/    # Ground truth segmentation masks
    └── train_data_CFZ.csv      # Training metadata
    ```

### Training

1.  **Configure Training Parameters**:
    Adjust hyperparameters in [config.py](config.py):
    - `BATCH_SIZE`: Batch size for training (default: 15)
    - `IMAGESIZE`: Input image size (default: 320)
    - `FOLD_NUM` and `MAX_FOLD`: Cross-validation settings

2.  **Optional: Download Pretrained Weights**:
    For better performance, you can use pretrained ViT weights. Download the R50+ViT-B_16 weights and update the path in [train.py](train.py).

3.  **Start Training**:
    ```bash
    python train.py
    ```

    The trained model will be saved to the path specified in `SAVE_PATH` in [config.py](config.py).

### Using the Notebook

For an interactive training experience, you can use the Jupyter notebook:
```bash
jupyter notebook notebooks/TransUnet_for_CFZ_Segmentation.ipynb
```
This notebook contains the complete pipeline from data preparation to model evaluation.

## Project Structure

```
TransUnet-for-CFZ-Segmentation/
├── config.py                 # Configuration and hyperparameters
├── model.py                  # TransUnet model implementation
├── data_utils.py            # Data loading and preprocessing utilities
├── data_preparation.py      # Dataset preparation script
├── train.py                 # Training script
├── requirements.txt         # Python dependencies
├── README.md               # Project documentation
├── LICENSE.txt             # License information
├── .gitignore             # Git ignore file
├── Figures/               # Sample segmentation results
└── notebooks/             # Jupyter notebooks
    └── TransUnet_for_CFZ_Segmentation.ipynb
```

## Citation

If you use this code in your research, please cite the TransUNet paper:

```bibtex
@article{chen2021transunet,
  title={TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation},
  author={Chen, Jieneng and Lu, Yongyi and Yu, Qihang and Luo, Xiangde and Adeli, Ehsan and Wang, Yan and Lu, Le and Yuille, Alan L and Zhou, Yuyin},
  journal={arXiv preprint arXiv:2102.04306},
  year={2021}
}
```

## License

This project is licensed under the terms specified in [LICENSE.txt](LICENSE.txt).

## Acknowledgments

- TransUNet architecture based on the original implementation
- OCTA images acquired using AngioVue SD-OCT device (Optovue, Fremont, CA, USA)
- Built with PyTorch and segmentation_models_pytorch
