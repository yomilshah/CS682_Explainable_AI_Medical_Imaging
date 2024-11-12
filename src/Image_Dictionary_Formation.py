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

def create_dictionary(hashes, only_duplicates=False):
    dict = {"filenames": [], "hashvalues": []}
    for hash_value, paths in hashes.items():
        if only_duplicates and len(paths) <= 1:
            continue
        dict["filenames"].extend([os.path.basename(path) for path in paths])
        dict["hashvalues"].extend([hash_value] * len(paths))
    return dict

# Construct the argument parser and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--datasets", nargs='+', required=True,
                help="paths to input datasets")
ap.add_argument("-o", "--option", type=int, choices=[1, 2], required=True,
                help="1: Create dictionaries with duplicates only; 2: Create dictionaries with all the images")
args = vars(ap.parse_args())

# Dictionary to store duplicates for each dataset
all_dataset_results = {}

# Process each dataset individually
for dataset in args["datasets"]:
    print(f"[INFO] Processing dataset: {dataset}")
    image_paths = list(paths.list_images(dataset))

    print("[INFO] Computing image hashes...")
    hashes = compute_hashes_parallel(image_paths)

    only_duplicates = args["option"] == 1
    # Create and store the duplicate dictionary for the current dataset
    result = create_dictionary(hashes, only_duplicates)
    
    all_dataset_results[os.path.basename(dataset)] = result 

# Display duplicate dictionaries for each dataset
for dataset_name, result in all_dataset_results.items():
        print(f"[INFO] Result for dataset '{dataset_name}':")
        print(result)
