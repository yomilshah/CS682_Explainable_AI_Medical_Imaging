from imutils import paths
import numpy as np
import cv2
import os
import concurrent.futures

def dhash(image, hashSize=8):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hashSize + 1, hashSize))
    diff = resized[:, 1:] > resized[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def process_image(imagePath):
    # load the input image and compute the hash
    image = cv2.imread(imagePath)
    if image is None:
        return None, None
    h = dhash(image)
    return h, imagePath

def compute_hashes_parallel(imagePaths):
    # Use ThreadPoolExecutor to compute image hashes in parallel
    hashes = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_image, imagePaths))

    # Build the hashes dictionary from the parallel results
    for h, imagePath in results:
        if h is not None:
            p = hashes.get(h, [])
            p.append(imagePath)
            hashes[h] = p
    return hashes

# construct the argument parser and parse the arguments
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-d1", "--dataset1", required=True,
                help="path to input dataset 1")
ap.add_argument("-d2", "--dataset2", required=True,
                help="path to input dataset 2")
ap.add_argument("-o", "--output", required=True,
                help="path to output text file to save duplicates")
args = vars(ap.parse_args())

# grab the paths to all images in both datasets
print("[INFO] computing image hashes for dataset 1...")
imagePaths1 = list(paths.list_images(args["dataset1"]))

print("[INFO] computing image hashes for dataset 2...")
imagePaths2 = list(paths.list_images(args["dataset2"]))

# compute the hashes in parallel for both datasets
hashes1 = compute_hashes_parallel(imagePaths1)
hashes2 = compute_hashes_parallel(imagePaths2)

# find duplicates between dataset_1 and dataset_2
duplicates = []
for (h1, hashedPaths1) in hashes1.items():
    if h1 in hashes2:
        hashedPaths2 = hashes2[h1]
        for p1 in hashedPaths1:
            for p2 in hashedPaths2:
                duplicates.append((p1, p2, h1))  # Add hash value to duplicates

# write the duplicate pairs with hash values to a text file
with open(args["output"], "w") as f:
    if duplicates:
        f.write("[INFO] Duplicate images found between the two datasets:\n")
        for (image1, image2, hashValue) in duplicates:
            f.write(f"{os.path.basename(image1)} <--> {os.path.basename(image2)}  {hashValue}\n")
        print(f"[INFO] Duplicate image pairs written to: {args['output']}")
    else:
        f.write("[INFO] No duplicates found between the two datasets.\n")
        print("[INFO] No duplicates found between the two datasets.")