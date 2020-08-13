import numpy as np
import glob
import os
import keras
from PIL import Image
import shutil
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import numpy.ma as ma
import os
import shutil
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, Dense, Dropout, Activation, Flatten, BatchNormalization, Reshape
from keras.engine.topology import Layer
from keras.layers.merge import concatenate, add
from keras.layers.core import Lambda
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
import imageio
from natsort import natsorted
import random
import warnings
import segmentation_models as sm
import sys
warnings.filterwarnings("ignore")

location = sys.argv[1]
model_path = sys.argv[2]
regex_images = '%s/images/all_images/*.png' % location
images = glob.glob(regex_images)
patients = [(p.split('all_images/')[1]).split('_')[0] for p in images]
patients = set(patients)
patients = list(patients)

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


# load the pre-trained neural network weights
model = load_model(model_path, custom_objects={'dice_loss': dice_loss, 'dice_coeff': dice_coeff})
model.compile(optimizer = Adam(lr = 1e-4), loss=dice_loss, metrics=[dice_coeff])

def np_dice(true, pred):
    intersection = np.sum(true * pred)
    dc =(2.0 * intersection) / (np.sum(true) + np.sum(pred))
    return dc

dice_scores = []

for patient in patients:

    regex_images = '%s/images/all_images/%s*.png' % (location, patient)
    images = glob.glob(regex_images)
    scans = [(p.split('all_images/')[1]).split('_img')[0] for p in images]
    scans = list(set(scans))

    for scan in scans:

        regex_images = '%s/images/all_images/%s*.png' % (location, scan)
        images = glob.glob(regex_images)
        masks = [p.replace('/images/', '/masks/') for p in images]

        X_test_seg = np.zeros((len(images), 256, 256, 1))
        Y_test_seg = np.zeros((len(images), 256, 256, 1))

        for i in range(0, len(images)):
            img = np.asarray(Image.open(images[i]))
            mask = np.asarray(Image.open(masks[i]))
            X_test_seg[i,:,:,0] = adjust_sizes(img)
            Y_test_seg[i,:,:,0] = adjust_sizes(mask)

        X_test_seg = X_test_seg/255
        Y_test_seg = Y_test_seg/255

        preds_test = model.predict(X_test_seg, verbose=1)
        preds_test = (preds_test > 0.5).astype(np.uint8)

        dsc = np_dice(Y_test_seg, preds_test)
        dice_scores.append(dsc)

dice_score = sum(dice_scores)/len(dice_scores)

print("The dice score for this model is: ", dice_score)