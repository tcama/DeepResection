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
import pandas as pd
import matplotlib.pyplot as plt

# define arguments and load data
IMG_DIR = sys.argv[1]
MASK_DIR = sys.argv[2]
ATLAS_DIR = sys.argv[3]
ATLAS_MAPPINGS = sys.argv[4]
OUTPUT_DIR = sys.argv[5]
OUTPUT_DIR = os.path.join(OUTPUT_DIR, "resected_results.txt")

mask = nib.load(MASK_DIR)
mask_data = mask.get_fdata()

img = nib.load(IMG_DIR)
img_data = img.get_fdata()

atlas = nib.load(ATLAS_DIR)
atlas_data = atlas.get_fdata()

# multiply the arrays to get ROIs of the regions contained in the resection zone
combined = mask_data * atlas_data

# get each ROI value in the result
roi_values = np.unique(combined)

roi_values = list(roi_values)
roi_values = roi_values[1:]

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

# initialize the list of results to be printed to the 
results = []
volume_str = "Total resection volume (cubic cm): " + str(resection_volume)
results.append(volume_str)

# loop through each ROI, perform calculations, and add entry to the list of results
df = []
for roi in roi_values:
    name = mappings[int(roi)]
    total = np.sum(atlas_data == roi)
    resected = np.sum(combined == roi)
    percentage = (resected/total) * 100
    percentage = np.round(percentage, 3)
    percentage_remaining = 100 - percentage
    volume = resected * new_dims[0] * new_dims[1] * new_dims[2]
    result_str = name + ": " + str(percentage_remaining) + "% remaining"
    results.append(result_str)
    df.append({'Region': name, 'Remaining (%)': percentage_remaining, 'Volume Resected (cubic cm)': volume})

# print and save results
print('Resection results, by region:')
output_str = ""
for result in results:
    print(result)
    output_str = output_str + result + "\n"

with open(OUTPUT_DIR, "w") as f:
    f.write(output_str)

# save PDF
HTML_DIR = os.path.join(sys.argv[5], "resection_report.html")
df = pd.DataFrame(df)
df.to_html(HTML_DIR)

# figure for output
fig, ax = plt.subplots(3,2)

#[x,y,z] = np.shape(mask_data)
mask_data = np.ma.masked_where(mask_data == 0, mask_data)
x = np.argmax( np.sum(mask_data, axis=(1,2) ) )
y = np.argmax( np.sum(mask_data, axis=(0,2) ) )
z = np.argmax( np.sum(mask_data, axis=(0,1) ) )

# Sagittal
fig.add_subplot(3,2,1)
plt.imshow(np.rot90(img_data[x,:,:]), cmap = "gray")
plt.clim(np.min(img_data), np.max(img_data))
plt.axis('off')
fig.add_subplot(3,2,2)
plt.imshow(np.rot90(img_data[x,:,:]), cmap = "gray")
plt.clim(np.min(img_data), np.max(img_data))
plt.imshow(np.rot90(mask_data[x,:,:]), 'cool', alpha=0.4)
plt.axis('off')

# Coronal
fig.add_subplot(3,2,3)
plt.imshow(np.rot90(img_data[:,y,:]), cmap = "gray")
plt.clim(np.min(img_data), np.max(img_data))
plt.axis('off')
fig.add_subplot(3,2,4)
plt.imshow(np.rot90(img_data[:,y,:]), cmap = "gray")
plt.clim(np.min(img_data), np.max(img_data))
plt.imshow(np.rot90(mask_data[:,y,:]), 'cool', alpha=0.4)
plt.axis('off')

# Axial
fig.add_subplot(3,2,5)
plt.imshow(img_data[:,:,z], cmap = "gray")
plt.clim(np.min(img_data), np.max(img_data))
plt.axis('off')
fig.add_subplot(3,2,6)
plt.imshow(img_data[:,:,z], cmap = "gray")
plt.clim(np.min(img_data), np.max(img_data))
plt.imshow(mask_data[:,:,z], 'cool', alpha=0.4)
plt.axis('off')

[axi.set_axis_off() for axi in ax.ravel()]
fig.set_size_inches(10, 6)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=-0.5, hspace=0.1)
plt.tight_layout()

FIG_DIR = os.path.join(sys.argv[5], "resection_views.png")
fig.savefig(FIG_DIR)

with open(HTML_DIR, "a") as myfile:
    myfile.write( "\n <img src=\"" + "resection_views.png" + "\" align=\"top right\"/>" )
