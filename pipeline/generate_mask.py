# This code uses the trained deep learning model to generate a mask NIFTI file given the NIFTI file for the post-operative MRI
# For now I am using the U-Net with VGG16 backbone built using the segmentation_models codebase
# Only run after installing the codebase: pip3 install segmentation_models

# Usage: generate_mask.py postop.nii out_dir out_name
# Example: generate_mask.py patient1_postop.nii ./patient1_info patient1_mask.nii

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
        output_file_name = PATIENT_NAME + '_img00' + str(slice) + '_postop.png'
    elif slice < 100:
        output_file_name = PATIENT_NAME + '_img0' + str(slice) + '_postop.png'
    else:
        output_file_name = PATIENT_NAME + '_img' + str(slice) + '_postop.png'
    
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

# build the model architecture
BACKBONE = 'vgg16'

base_model = sm.Unet(BACKBONE, encoder_weights='imagenet', classes=1, activation='sigmoid')
inp = Input(shape=(256, 256, 1))
l1 = Conv2D(3, (1, 1)) (inp)
out = base_model(l1)
model = Model(inp, out, name = base_model.name)
model.compile(optimizer = Adam(lr = 1e-4), loss=dice_loss, metrics=[dice_coeff])

# load the pre-trained neural network weights
model.load_weights('../analysis/model_vgg16.h5')

# predict the resected tissue for each slice in the 3D input array
preds = model.predict(input_arr, verbose = 1)
preds = (preds > 0.5).astype(np.uint8)

# adjust the output array dimensions so that they are the same as the original 3D image array
realX = postop_3D.shape[0]
realY = postop_3D.shape[1]
output = np.zeros(postop_3D.shape)
for slice in range(0, total_slices):
    output[:,:,slice] = adjust_sizes(preds[slice,:,:,0], dim = (realX, realY))

# convert the output array into a NIFTI file
ni_mask = nib.Nifti1Image(output, postop.affine)
OUTPUT_DIR = sys.argv[2]
MASK_OUTPUT_FILE = os.path.join(OUTPUT_DIR, sys.argv[3])
nib.save(ni_mask, MASK_OUTPUT_FILE)