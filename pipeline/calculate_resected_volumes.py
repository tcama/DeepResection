# This code uses the given mask and pre-operative atlas (registered to the post-operative image) NIFTI files to calculate
# and print percent volume resected for each ROI in the brain

# Usage: calculate_resected_volumes.py mask.nii atlas.nii output_dir
# Example: calculate_resected_volumes.py masks/patient1_mask.nii atlases/patient1_AAL116_origin_MNI_T1.nii results/patient1

# 6/4/20 - created

import sys
import numpy as np
import nibabel as nib
import imageio
import os
import json

# define arguments and load data
MASK_DIR = sys.argv[1]
ATLAS_DIR = sys.argv[2]
OUTPUT_DIR = sys.argv[3]
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

# define the mappings from atlas ROI pixel value to ROI name
mappings = {2001: "Precentral gyrus left", 2002: "Precentral gyrus right", 2101: "Superior frontal gyrus, dorsolateral left",
            2102: "Superior frontal gyrus, dorsolateral right", 2111: "Superior frontal gyrus, orbital left",
            2112: "Superior frontal gyrus, orbital right", 2201: "Middle frontal gyrus left", 2202: "Middle frontal gyrus right",
            2211: "Middle frontal gyrus, orbital left", 2212: "Middle frontal gyrus, orbital right",
            2301: "Inferior frontal gyrus, opercular left", 2302: "Inferior frontal gyrus, opercular right",
            2311: "Inferior frontal gyrus, triangular left", 2312: "Inferior frontal gyrus, triangular right",
            2321: "Inferior frontal gyrus, orbital left", 2322: "Inferior frontal gyrus, orbital right", 2331: "Rolandic operculum left",
            2332: "Rolandic operculum right", 2401: "Supplementary motor area left", 2402: "Supplementary motor area right",
            2501: "Olfactory cortex left", 2502: "Olfactory cortex right", 2601: "Superior frontal gyrus, medial left",
            2602: "Superior frontal gyrus, medial right", 2611: "Superior frontal gyrus, medial orbital left",
            2612: "Superior frontal gyrus, medial orbital right", 2701: "Gyrus rectus left", 2702: "Gyrus rectus right",
            3001: "Insula left", 3002: "Insula right", 4001: "Cingulate gyrus, anterior left", 4002: "Cingulate gyrus, anterior right",
            4011: "Cingulate gyrus, mid left", 4012: "Cingulate gyrus, mid right", 4021: "Cingulate gyrus, posterior left",
            4022: "Cingulate gyrus, posterior right", 4101: "Hippocampus left", 4102: "Hippocampus right", 4111: "Parahippocampus left",
            4112: "Parahippocampus right", 4201: "Amygdala left", 4202: "Amygdala right", 5001: "Calcarine fissure/cortex left",
            5002: "Calcarine fissure/cortex right", 5011: "Cuneus left", 5012: "Cuneus right", 5021: "Lingual gyrus left",
            5022: "Lingual gyrus right", 5101: "Superior occipital lobe left", 5102: "Superior occipital lobe right",
            5201: "Middle occipital lobe left", 5202: "Middle occipital lobe right", 5301: "Inferior occipital lobe left",
            5302: "Inferior occipital lobe right", 5401: "Fusiform gyrus left", 5402: "Fusiform gyrus right",
            6001: "Postcentral gyrus left", 6002: "Postcentral gyrus right", 6101: "Superior parietal gyrus left",
            6102: "Superior parietal gyrus right", 6201: "Inferior parietal gyrus left", 6202: "Inferior parietal gyrus right",
            6211: "Supramarginal gyrus left", 6212: "Supramarginal gyrus right", 6221: "Angular gyrus left", 6222: "Angular gyrus right",
            6301: "Precuneus left", 6302: "Precuneus right", 6401: "Paracentral lobule left", 6402: "Paracentral lobule right",
            7001: "Caudate nucleus left", 7002: "Caudate nucleus right", 7011: "Putamen left", 7012: "Putamen right",
            7021: "Pallidum left", 7022: "Pallidum right", 7101: "Thalamus left", 7102: "Thalamus right",
            8101: "Heschl gyrus left", 8102: "Heschl gyrus right", 8111: "Superior temporal gyrus left",
            8112: "Superior temporal gyrus right", 8121: "Temporal pole: superior temporal gyrus left",
            8122: "Temporal pole: superior temporal gyrus right", 8201: "Middle temporal gyrus left", 8202: "Middle temporal gyrus right",
            8211: "Temporal pole: middle temporal gyrus left", 8212: "Temporal pole: middle temporal gyrus right",
            8301: "Inferior temporal gyrus left", 8302: "Inferior temporal gyrus right", 9001: "Cerebellum crus 1 left",
            9002: "Cerebellum crus 1 right", 9011: "Cerebellum crus 2 left", 9012: "Cerebellum crus 2 right",
            9021: "Cerebellum 3 left", 9022: "Cerebellum 3 right", 9031: "Cerebellum 4, 5 left", 9032: "Cerebellum 4, 5 right",
            9041: "Cerebellum 6 left", 9042: "Cerebellum 6 right", 9051: "Cerebellum 7 left", 9052: "Cerebellum 7 right",
            9061: "Cerebellum 8 left", 9062: "Cerebellum 8 right", 9071: "Cerebellum 9 left", 9072: "Cerebellum 9 right",
            9081: "Cerebellum 10 left", 9082: "Cerebellum 10 right", 9100: "Vermis 1, 2", 9110: "Vermis 3", 9120: "Vermis 4, 5",
            9130: "Vermis 6", 9140: "Vermis 7", 9150: "Vermis 8", 9160: "Vermis 9", 9170: "Vermis 10"}

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

with open(OUTPUT_DIR, 'w') as f:
    f.write(output_str)