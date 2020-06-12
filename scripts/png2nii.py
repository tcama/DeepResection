# This code takes a directory with png files representing horizontal slices of a 3D image, an existing directory to save the output in, a name
# for the output file, and a nifti file whose size & reference space are matched by the output
# It outputs a .nii file for the entire 3D image

# The input directory must not contain any other files, and the images in the input directory must be labeled such that they are naturally sorted in order of slice

# Usage: png2nii.py input_dir output_dir output_name affine.nii

# Example: png2nii.py data/png/25_f/img analysis/25_f 25_f_predicted_mask.nii data/25_f/25_f_postop.nii

import numpy as np
import nibabel as nib
import imageio
import os
import sys
from natsort import natsorted

# function that pads or crops an array (in numpy format) to a specified dimension
def adjust_sizes(img_arr, dim = (256, 256)):
    if(img_arr.shape[0] < dim[0]):
        toBeAdded = dim[0] - img_arr.shape[0]
        toBeAdded1 = int(np.ceil(toBeAdded/2))
        toBeAdded2 = int(np.floor(toBeAdded/2))
        img_arr = np.pad(img_arr, ((toBeAdded1, toBeAdded2), (0, 0)))
    elif(img_arr.shape[0] > dim[0]):
        toBeRemoved = img_arr.shape[0]-dim[0]
        toBeRemoved1 = int(np.ceil(toBeRemoved/2))
        toBeRemoved2 = int(np.floor(toBeRemoved/2))
        index1 = toBeRemoved1
        index2 = img_arr.shape[0]-toBeRemoved2
        img_arr = img_arr[index1:index2,:]
    if(img_arr.shape[1] < dim[1]):
        toBeAdded = dim[1] - img_arr.shape[1]
        toBeAdded1 = int(np.ceil(toBeAdded/2))
        toBeAdded2 = int(np.floor(toBeAdded/2))
        img_arr = np.pad(img_arr, ((0, 0), (toBeAdded1, toBeAdded2)))
    elif(img_arr.shape[1] > dim[1]):
        toBeRemoved = img_arr.shape[1]-dim[1]
        toBeRemoved1 = int(np.ceil(toBeRemoved/2))
        toBeRemoved2 = int(np.floor(toBeRemoved/2))
        index1 = toBeRemoved1
        index2 = img_arr.shape[1]-toBeRemoved2
        img_arr = img_arr[:,index1:index2]
    return img_arr

# define inputs and check for errors
if(len(sys.argv) != 5):
    print("Incorrect number of arguments")
    sys.exit(1)

INPUT_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
OUTPUT_NAME = sys.argv[3]
AFFINE_INP = sys.argv[4]

affine_ni = nib.load(AFFINE_INP)
affine_data = affine_ni.get_fdata()

img_list = natsorted(os.listdir(INPUT_DIR))
print(img_list)

slices = len(img_list)
if(slices != affine_data.shape[2]):
    print("Affine nifti file does not have the same number of slices as the png inputs")
    sys.exit(1)

output = np.zeros(affine_data.shape)
for img_idx in range(slices):
    img_file = os.path.join(INPUT_DIR, img_list[img_idx])
    img_arr = np.asarray(imageio.imread(img_file))
    output[:,:,img_idx] = adjust_sizes(img_arr, dim = (output.shape[0], output.shape[1]))

output_file = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
output_ni = nib.Nifti1Image(output, affine_ni.affine)
nib.save(output_ni, output_file)
    
