# data_preparation.py

import os
import cv2
import numpy as np
import pandas as pd
from shutil import copyfile

# Import configurations
from config import (
    IMAGESIZE, SOURCE_DIR, DATASET_PATH, TRAIN_PATH, INPUT_PATH,
    OUTPUT_PATH, PARENT_DIR, DATAFRAME_NAME, SAVE_PATH, PROJECT_PATH
)

def create_directories():
    """Creates the necessary directory structure for the project."""
    dirs_to_create = [
        SAVE_PATH, PROJECT_PATH, DATASET_PATH, TRAIN_PATH, INPUT_PATH, OUTPUT_PATH
    ]
    for d in dirs_to_create:
        try:
            os.makedirs(d, exist_ok=True) # exist_ok=True prevents error if dir already exists
        except OSError as e:
            print(f"Error creating directory {d}: {e}")

def tranfer_data(SOURCE, DESTINATION, imagesize):
    """Transfers and processes image files."""
    files = []
    # Ensure destination directories exist
    os.makedirs(os.path.join(DESTINATION, 'train/Input'), exist_ok=True)
    os.makedirs(os.path.join(DESTINATION, 'train/CFZ_map'), exist_ok=True)

    for subfolder in os.listdir(SOURCE):
        subfolder_path = os.path.join(SOURCE, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        itnum = 6 if subfolder == 'NPDR_Mild' else 1
        
        ava_map_path = os.path.join(subfolder_path, 'AVA_map')
        octa_path = os.path.join(subfolder_path, 'OCTA')

        if not os.path.exists(ava_map_path) or not os.path.exists(octa_path):
            print(f"Skipping {subfolder}: missing AVA_map or OCTA directory.")
            continue

        for ite in range(itnum):
            for filename in os.listdir(ava_map_path):
                if filename.endswith('_CFZ_MM.png'):
                    base_filename = filename[:-11]

                    file_octa_tiff = os.path.join(octa_path, base_filename + '.tiff')
                    file_octa_png = os.path.join(octa_path, base_filename + '.png')
                    file_cfz_map = os.path.join(ava_map_path, filename)

                    file_octa = None
                    if os.path.isfile(file_octa_tiff):
                        file_octa = file_octa_tiff
                    elif os.path.isfile(file_octa_png):
                        file_octa = file_octa_png
                    else:
                        print(f"OCTA image not found for {base_filename} in {subfolder_path}")
                        continue
                    
                    if not os.path.isfile(file_cfz_map):
                        print(f"CFZ map not found for {filename} in {subfolder_path}")
                        continue

                    # Define the destination paths
                    output_filename_base = filename[:-5] + '_' + str(ite)
                    destination_input = os.path.join(DESTINATION, 'train/Input', output_filename_base + '.png')
                    destination_output = os.path.join(DESTINATION, 'train/CFZ_map', output_filename_base + '.png')

                    # Read and resize the OCTA and CFZ map images
                    im_octa = cv2.imread(file_octa, cv2.IMREAD_GRAYSCALE)
                    im_cfz_map = cv2.imread(file_cfz_map)

                    if im_octa is None or im_cfz_map is None:
                        print(f"Failed to read image for {base_filename} in {subfolder_path}")
                        continue

                    im_octa = cv2.resize(im_octa, (imagesize, imagesize))
                    im_cfz_map = cv2.resize(im_cfz_map, (imagesize, imagesize))

                    # Normalize OCTA images to 0-255 scale
                    im_octa = np.array(im_octa, dtype=np.float32)
                    im_cfz_map = np.array(im_cfz_map, dtype=np.float32)
                    im_octa = 255.0 * (im_octa - np.min(im_octa)) / (np.max(im_octa) - np.min(im_octa))

                    # Convert grayscale image to 3-channel
                    im_octa_3chs = cv2.merge([im_octa, im_octa, im_octa])

                    # Save the processed images
                    cv2.imwrite(destination_input, im_octa_3chs.astype(np.uint8))
                    cv2.imwrite(destination_output, im_cfz_map.astype(np.uint8))

                    files.append(output_filename_base + '.png')
    
    files = list(set(files)) # Remove duplicates if any
    return files

if __name__ == '__main__':
    create_directories()
    print("Directories created.")

    print(f"Starting data transfer from {SOURCE_DIR} to {DATASET_PATH}...")
    filenames = tranfer_data(SOURCE_DIR, DATASET_PATH, IMAGESIZE)
    print(f"Transferred {len(filenames)} files.")

    # Create the DataFrame and save to CSV
    dframe1 = pd.DataFrame({'Input': filenames, 'CFZ': filenames})
    csv_save_path = os.path.join(PARENT_DIR, DATAFRAME_NAME)
    dframe1.to_csv(csv_save_path, index=False)
    print(f"DataFrame saved to {csv_save_path}")
    print(dframe1.head())