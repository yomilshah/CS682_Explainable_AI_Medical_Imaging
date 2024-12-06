from imutils import paths
import numpy as np
import argparse
import cv2
import os
import concurrent.futures
import json


# Function to compute dhash of an image
def dhash(image, hashSize=8):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hashSize + 1, hashSize))
    diff = resized[:, 1:] > resized[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


# Process a single image and return its hash
def process_image(imagePath):
    image = cv2.imread(imagePath)
    if image is None:
        return None, None
    h = dhash(image)
    return h, imagePath


# Compute hashes for a list of image paths in parallel
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

# Generate nested dictionary for option 3
def find_duplicates_nested(all_hashes):
    nested_dict = {"dataset": [], "hashvalues": [], "pair": []}

    # Combine all hashes across datasets
    hash_to_images = {}
    for dataset_path, hashes in all_hashes.items():
        for h, paths in hashes.items():
            hash_to_images.setdefault(h, []).extend(paths)

    # Process each hash to build the nested structure
    for hash_value, paths in hash_to_images.items():
        if len(paths) > 1:
            # Add all dataset:image entries for this hash
            for image_path in paths:
                dataset = os.path.basename(os.path.dirname(image_path))
                filename = os.path.basename(image_path)
                nested_dict["dataset"].append(f"{dataset}: {{{filename}}}")
                nested_dict["hashvalues"].append(hash_value)

            # Add duplicate pairs for each image
            for primary in paths:
                primary_dataset = os.path.basename(os.path.dirname(primary))
                primary_filename = os.path.basename(primary)
                primary_entry = f"{primary_dataset}: {{{primary_filename}}}"

                duplicate_dict = {}
                for duplicate in paths:
                    if duplicate != primary:
                        dup_dataset = os.path.basename(os.path.dirname(duplicate))
                        dup_filename = os.path.basename(duplicate)
                        duplicate_dict[dup_dataset] = f"{{{dup_filename}}}"

                nested_dict["pair"].append(duplicate_dict)

    return nested_dict


# Construct the argument parser and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--datasets", nargs='+', required=True,
                help="paths to input datasets")
ap.add_argument("-o", "--option", type=int, choices=[1, 2, 3], required=True,
                help="1: Create dictionaries with duplicates only; 2: Create dictionaries with all the images; 3: Duplicates across datasets")
ap.add_argument("-j", "--json", required=True,
                help="path to save the output JSON file")
args = vars(ap.parse_args())

# Dictionary to store hashes for each dataset
all_dataset_results = {}
combined_hashes = {}
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
# Process option 3
elif args["option"] == 3:
    # Combine hashes across datasets
    for dataset in args["datasets"]:
        print(f"[INFO] Processing dataset: {dataset}")
        image_paths = list(paths.list_images(dataset))
        all_hashes[dataset] = compute_hashes_parallel(image_paths)

    # Generate the nested dictionary
    result = find_duplicates_nested(all_hashes)

    # Save the result to JSON
    # output_file = args["json"]
    # with open(output_file, "w") as f:
    #     json.dump(result, f, indent=4)

    # print(f"[INFO] Nested dictionary saved to {output_file}")
output_file = args["json"]
if args["option"] in [1, 2]:
    with open(output_file, "w") as f:
        json.dump(all_dataset_results, f, indent=4)
elif args["option"] == 3:
    with open(output_file, "w") as f:
        json.dump(result, f, indent=4)
    print(f"[INFO] Nested dictionary saved to {output_file}")
