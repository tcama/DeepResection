# This code uses the trained deep learning model to generate a mask NIFTI file given the NIFTI file for the post-operative MRI
# For now I am using the U-Net with VGG16 backbone built using the segmentation_models codebase
# Only run after installing the codebase: pip3 install segmentation_models

# Usage: generate_mask.py postop.nii out_dir out_name is_continuous
# Example: generate_mask.py patient1_postop.nii ./patient1_info patient1_mask.nii True

# 6/3/20 - created

import sys
import numpy as np
import nibabel as nib
import imageio
import os
import shutil
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Conv2D
from keras.optimizers import Adam
from keras.utils.generic_utils import get_custom_objects
import warnings
from skimage.measure import label
from radiomics.shape import RadiomicsShape
from radiomics.featureextractor import RadiomicsFeatureExtractor
from sklearn.cluster import KMeans
warnings.filterwarnings("ignore")


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

# load the post-operative file and convert to a numpy array
POSTOP_FILE = sys.argv[1]
postop = nib.load(POSTOP_FILE)
postop_3D = postop.get_fdata()

# number of slices in the 3D image
total_slices = postop_3D.shape[2]

# initialize the input array that will be fed into the segmentation model
input_arr = np.zeros((total_slices, 256, 256, 1))

MAX_IMG = np.max(postop_3D)

# loop through all slices
for slice in range(0, total_slices):
    if slice < 10:
        output_file_name = 'temp_img00' + str(slice) + '_postop.png'
    elif slice < 100:
        output_file_name = 'temp_img0' + str(slice) + '_postop.png'
    else:
        output_file_name = 'temp_img' + str(slice) + '_postop.png'
    
    # normalize the pixel values of the slice
    img = postop_3D[:,:,slice]/MAX_IMG

    # save as a png file
    imageio.imwrite(output_file_name, img)

    # reload the image
    img_arr = np.asarray(imageio.imread(output_file_name))

    # adjust dimensions of image
    img_adj = adjust_sizes(img_arr)

    # update the input array
    input_arr[slice,:,:,0] = img_adj

    # remove the png file for memory purposes
    os.remove(output_file_name)

# rescale pixel values
input_arr = input_arr / 255.0

# define loss measure for the model so that it will compile
def dice_coeff(y_true, y_pred):
    epsilon = 10 ** -7
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true + y_pred)
    dice_score = (2.0 * intersection + epsilon) / (union + epsilon)
    return dice_score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

get_custom_objects().update({"dice": dice_loss})


# load the pre-trained neural network weights
model = load_model('analysis/model_inception.h5', custom_objects={'dice_loss': dice_loss, 'dice_coeff': dice_coeff})

# predict the resected tissue for each slice in the 3D input array
preds = model.predict(input_arr, verbose = 1)
preds = (preds > 0.5).astype(np.uint8)

# adjust the output array dimensions so that they are the same as the original 3D image array
realX = postop_3D.shape[0]
realY = postop_3D.shape[1]
output = np.zeros(postop_3D.shape)
for slice in range(0, total_slices):
    output[:,:,slice] = adjust_sizes(preds[slice,:,:,0], dim = (realX, realY))

# get each continuous object in the mask
IS_CONTINUOUS = sys.argv[4]
if IS_CONTINUOUS:
    mask_data_int = output.astype(np.uint8)
    labels =label(mask_data_int, connectivity = 2)
    object_labels = list(np.unique(labels))
    object_labels.remove(0)

    # get the continuous object that has the largest volume
    MAX_OBJ = 0
    MAX_OBJ_VOL = 0
    for obj_lab in object_labels:
        vol = np.sum(labels == obj_lab)
        if vol > MAX_OBJ_VOL:
            MAX_OBJ_VOL = vol
            MAX_OBJ = obj_lab

    # remove every other object from the mask
    if MAX_OBJ != 0:
        object_labels.remove(MAX_OBJ)
    for obj_lab in object_labels:
        idx = (labels == obj_lab)
        output[idx] = 0

# convert the output array into a NIFTI file
ni_mask = nib.Nifti1Image(output, postop.affine)
OUTPUT_DIR = sys.argv[2]
MASK_OUTPUT_FILE = os.path.join(OUTPUT_DIR, sys.argv[3])
nib.save(ni_mask, MASK_OUTPUT_FILE)
