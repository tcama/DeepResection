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
warnings.filterwarnings("ignore")

seed = 909
test_datagen_args = dict(rescale = 1./255)
image_datagen_test = ImageDataGenerator(**test_datagen_args)
mask_datagen_test = ImageDataGenerator(**test_datagen_args)
image_generator_test = image_datagen_test.flow_from_directory('test_3d/images', seed = seed, batch_size=1, target_size = (256, 256), classes=None, class_mode=None, color_mode = 'grayscale')
mask_generator_test = mask_datagen_test.flow_from_directory('test_3d/masks', seed = seed, batch_size=1, target_size = (256, 256), classes = None, class_mode=None, color_mode = 'grayscale')
test_generator = (pair for pair in zip(image_generator_test, mask_generator_test))

num_test_samples = len(os.listdir('test_3d/images/all_images'))

X_test_seg = np.zeros((num_test_samples, 256, 256, 1))
Y_test_seg = np.zeros((num_test_samples, 256, 256, 1))
for step in range(0, num_test_samples):
    X_batch, Y_batch = next(test_generator)
    X_test_seg[step,:,:,:] = X_batch[0,:,:,:]
    Y_test_seg[step,:,:,:] = np.round(Y_batch[0,:,:,:])

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
model = load_model('model_inception_3d.h5', custom_objects={'dice_loss': dice_loss, 'dice_coeff': dice_coeff})
model.compile(optimizer = Adam(lr = 1e-4), loss=dice_loss, metrics=[dice_coeff])

preds_test = model.predict(X_test_seg, verbose=1)
preds_test = (preds_test > 0.5).astype(np.uint8)

def np_dice(true, pred):
    intersection = np.sum(true * pred)
    dc =(2.0 * intersection) / (np.sum(true) + np.sum(pred))
    return dc

print("The dice score for this model is: ", np_dice(Y_test_seg, preds_test))