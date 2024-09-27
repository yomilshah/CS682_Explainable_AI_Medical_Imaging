from imutils import paths
import numpy as np
import argparse
import cv2
import os
import concurrent.futures

def dhash(image, hashSize=8):
	# convert the image to grayscale and resize the grayscale image,
	# adding a single column (width) so we can compute the horizontal gradient
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	resized = cv2.resize(gray, (hashSize + 1, hashSize))
	# compute the (relative) horizontal gradient between adjacent column pixels
	diff = resized[:, 1:] > resized[:, :-1]
	# convert the difference image to a hash and return it
	return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def process_image(imagePath):
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
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-r", "--remove", type=int, default=-1,
	help="whether or not duplicates should be removed (i.e., dry run)")
args = vars(ap.parse_args())


print("[INFO] computing image hashes...")
imagePaths = list(paths.list_images(args["dataset"]))
hashes = compute_hashes_parallel(imagePaths)

# loop over the image hashes
for (h, hashedPaths) in hashes.items():
	# check to see if there is more than one image with the same hash
	if len(hashedPaths) > 1:
		# check to see if this is a dry run
		if args["remove"] <= 0:
			print("[INFO] Duplicate images found for hash {}: ".format(h))
			# loop over all image paths with the same hash
			for p in hashedPaths:
				print(p)
				
		else:
			print(f"[INFO] Removing duplicates for hash {h}:")
			# loop over all image paths with the same hash except for the first image in the list
			for p in hashedPaths[1:]:
				os.remove(p)
