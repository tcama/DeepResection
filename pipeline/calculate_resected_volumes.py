# This code uses the given mask and pre-operative atlas (registered to the post-operative image) NIFTI files to calculate
# and print percent volume resected for each ROI in the brain

# Usage: calculate_resected_volumes.py mask.nii atlas.nii output_dir
# Example: calculate_resected_volumes.py masks/patient1_mask.nii atlases/patient1_AAL116_origin_MNI_T1.nii results/patient1

# 6/8/20 - created

import sys
import numpy as np
import nibabel as nib
import imageio
import os
import json

# define arguments and load data
MASK_DIR = sys.argv[1]
ATLAS_DIR = sys.argv[2]
ATLAS_MAPPINGS = sys.argv[3]
OUTPUT_DIR = sys.argv[4]
OUTPUT_DIR = os.path.join(OUTPUT_DIR, "resected_results.txt")

mask = nib.load(MASK_DIR)
mask_data = mask.get_fdata()

atlas = nib.load(ATLAS_DIR)
atlas_data = atlas.get_fdata()

# get individual voxel dimensions and convert to centimeters
header = mask.header
voxel_dims = header.get_zooms()
new_dims = (voxel_dims[0] / 10, voxel_dims[1] / 10, voxel_dims[2] / 10)

# find the volume of resection in cubic centimeters
mask_voxels = np.sum(mask_data)
resection_volume = mask_voxels * new_dims[0] * new_dims[1] * new_dims[2]

# open the atlas mappings file and define the mappings from atlas ROI pixel value to ROI name
mappings = {}
atlas_txt = open(ATLAS_MAPPINGS, "r")
for line in atlas_txt:
    entry = line.split()
    # find the key
    for e in entry:
        try:
            k = int(e)
        except:
            continue
    # find the value
    entry.remove(str(k))
    v = max(entry, key = len)
    mappings[k] = v

# multiply the arrays to get ROIs of the regions contained in the resection zone
combined = mask_data * atlas_data

# get each ROI value in the result
roi_values = np.unique(combined)

roi_values = list(roi_values)
roi_values = roi_values[1:]

# initialize the list of results to be printed to the 
results = []
volume_str = "Total resection volume (cubic cm): " + str(resection_volume)
results.append(volume_str)

# loop through each ROI, perform calculations, and add entry to the list of results
for roi in roi_values:
    name = mappings[int(roi)]
    total = np.sum(atlas_data == roi)
    resected = np.sum(combined == roi)
    percentage = (resected/total) * 100
    percentage = np.round(percentage, 3)
    percentage_remaining = 100 - percentage
    result_str = name + ": " + str(percentage_remaining) + "% remaining"
    results.append(result_str)

# print and save results

print('Resection results, by region:')
output_str = ""
for result in results:
    print(result)
    output_str = output_str + result + "\n"

with open(OUTPUT_DIR, "w") as f:
    f.write(output_str)