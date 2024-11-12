from imutils import paths
import numpy as np
import argparse
import cv2
import os
import concurrent.futures

def dhash(image, hashSize=8):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hashSize + 1, hashSize))
    diff = resized[:, 1:] > resized[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def process_image(imagePath):
    image = cv2.imread(imagePath)
    if image is None:
        return None, None
    h = dhash(image)
    return h, imagePath

def compute_hashes_parallel(imagePaths):
    hashes = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_image, imagePaths))

    for h, imagePath in results:
        if h is not None:
            hashes.setdefault(h, []).append(imagePath)
    return hashes

def create_duplicate_dictionary(hashes):
    duplicates = {"filenames": [], "hashvalues": []}
    for hash_value, paths in hashes.items():
        if len(paths) > 1:
            duplicates["filenames"].extend([os.path.basename(path) for path in paths])
            duplicates["hashvalues"].extend([hash_value] * len(paths))
    return duplicates

# Construct the argument parser and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--datasets", nargs='+', required=True,
                help="paths to input datasets")
args = vars(ap.parse_args())

# Dictionary to store duplicates for each dataset
all_duplicates = {}

# Process each dataset individually
for dataset in args["datasets"]:
    print(f"[INFO] Processing dataset: {dataset}")
    image_paths = list(paths.list_images(dataset))

    print("[INFO] Computing image hashes...")
    hashes = compute_hashes_parallel(image_paths)

    # Create and store the duplicate dictionary for the current dataset
    duplicates = create_duplicate_dictionary(hashes)
    all_duplicates[os.path.basename(dataset)] = duplicates

# Display duplicate dictionaries for each dataset
for dataset_name, duplicates in all_duplicates.items():
    if duplicates["filenames"]:
        print(f"[INFO] Duplicate image dictionary for dataset '{dataset_name}':")
        print(duplicates)
    else:
        print(f"[INFO] No duplicates found in dataset '{dataset_name}'.")