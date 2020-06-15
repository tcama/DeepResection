# This code takes a .nii file and outputs a .png file for each horizontal slice in the output directory
# as well as a 3D array containing pixel values for each slice

# Usage: nii2png.py patient_id input.nii img_output_dir

# Example: nii2png.py 25_f 25_f_postop.nii.gz analysis/25_f

# 6/11/20 - created

import numpy as np
import nibabel as nib
import imageio
import os
import sys

# error conditions
if(len(sys.argv) != 4):
    print("Number of arguments is incorrect")
    sys.exit(1)

# load the post-operative file and convert to a numpy array
NIFTI_FILE = sys.argv[2]
nifti_obj = nib.load(NIFTI_FILE)
nifti_data = nifti_obj.get_fdata()

# number of slices in the 3D image
total_slices = nifti_obj.shape[2]

PATIENT_ID = sys.argv[1]
OUTPUT_DIR = sys.argv[3]
MAX_IMG = np.max(nifti_data)

# loop through all slices
for slice in range(0, total_slices):
    if slice < 10:
        output_file_name = "%s_img00%d.png" % (PATIENT_ID, slice)
    elif slice < 100:
        output_file_name = "%s_img0%d.png" % (PATIENT_ID, slice)
    else:
        output_file_name = "%s_img%d.png" % (PATIENT_ID, slice)
    
    # normalize the pixel values of the slice
    img = nifti_data[:,:,slice]/MAX_IMG

    # save as a png file
    output_full_file = os.path.join(OUTPUT_DIR, output_file_name)
    imageio.imwrite(output_full_file, img)
