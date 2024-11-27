from imutils import paths
import numpy as np
import argparse
import cv2
import os
import concurrent.futures
import json

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
    if only_duplicates:
        dict = {"filenames": [], "hashvalues": [], "duplicates": []}
        for hash_value, paths in hashes.items():
            if len(paths) > 1:
                for i, primary in enumerate(paths):
                    duplicates = [os.path.basename(p) for j, p in enumerate(paths) if j != i]
                    dict["filenames"].append(os.path.basename(primary))
                    dict["hashvalues"].append(hash_value)
                    dict["duplicates"].append(", ".join(duplicates))
    else:
        dict = {"filenames": [], "hashvalues": []}
        for hash_value, paths in hashes.items():
            dict["filenames"].extend([os.path.basename(path) for path in paths])
            dict["hashvalues"].extend([hash_value] * len(paths))
    return dict

def find_duplicates_across_datasets(all_hashes):
    cross_dataset_dict = {"filenames": [], "hashvalues": [], "duplicates": []}

    hash_to_images = {}
    for dataset_name, hashes in all_hashes.items():
        for h, paths in hashes.items():
            hash_to_images.setdefault(h, []).extend(paths)

    for hash_value, paths in hash_to_images.items():
        if len(paths) > 1:
            for i, primary in enumerate(paths):
                duplicates = [p for j, p in enumerate(paths) if j != i]
                cross_dataset_dict["filenames"].append(primary)
                cross_dataset_dict["hashvalues"].append(hash_value)
                cross_dataset_dict["duplicates"].append(", ".join(duplicates))
    return cross_dataset_dict

# Construct the argument parser and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--datasets", nargs='+', required=True,
                help="paths to input datasets")
ap.add_argument("-o", "--option", type=int, choices=[1, 2, 3], required=True,
                help="1: Create dictionaries with duplicates only; 2: Create dictionaries with all the images; 3: Duplicates across datasets")
ap.add_argument("-j", "--json", required=True,
                help="path to save the output JSON file")
args = vars(ap.parse_args())

# Dictionary to store duplicates for each dataset
all_dataset_results = {}
all_hashes = {}

if args["option"] in [1, 2]:
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

elif args["option"] == 3:
    # Combine hashes across datasets
    for dataset in args["datasets"]:
        print(f"[INFO] Processing dataset: {dataset}")
        image_paths = list(paths.list_images(dataset))
        all_hashes[dataset] = compute_hashes_parallel(image_paths)

    result = find_duplicates_across_datasets(all_hashes)

# Display the results
output_file = args["json"]
if args["option"] in [1, 2]:
    with open(output_file, "w") as f:
        json.dump(all_dataset_results, f, indent=4)
    # for dataset_name, result in all_dataset_results.items():
    #     print(f"[INFO] Result for dataset '{dataset_name}':")
    #     print(result)
elif args["option"] == 3:
    with open(output_file, "w") as f:
        json.dump(result, f, indent=4)
    # print("[INFO] Result for duplicates across datasets:")
    # # print(all_dataset_results["cross_datasets"])
    # # print("[INFO] Duplicate image dictionary:")
    # print(result)
