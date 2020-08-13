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
datagen_args = dict(rescale = 1./255, rotation_range = 10, width_shift_range = 0.1, height_shift_range = 0.1, horizontal_flip = True, vertical_flip = True)
image_datagen = ImageDataGenerator(**datagen_args)
mask_datagen = ImageDataGenerator(**datagen_args)

image_generator = image_datagen.flow_from_directory('train_3d/images', seed = seed, batch_size=16, target_size = (256, 256), classes=None, class_mode=None, color_mode = 'grayscale')
mask_generator = mask_datagen.flow_from_directory('train_3d/masks', seed = seed, batch_size=16, target_size = (256, 256), classes = None, class_mode=None, color_mode = 'grayscale')
train_generator = (pair for pair in zip(image_generator, mask_generator))

num_train_samples = len(os.listdir('train_3d/images/all_images'))

test_datagen_args = dict(rescale = 1./255)
image_datagen_valid = ImageDataGenerator(**test_datagen_args)
mask_datagen_valid = ImageDataGenerator(**test_datagen_args)
image_generator_valid = image_datagen_valid.flow_from_directory('validation_3d/images', seed = seed, batch_size=1, target_size = (256, 256), classes=None, class_mode=None, color_mode = 'grayscale')
mask_generator_valid = mask_datagen_valid.flow_from_directory('validation_3d/masks', seed = seed, batch_size=1, target_size = (256, 256), classes = None, class_mode=None, color_mode = 'grayscale')
valid_generator = (pair for pair in zip(image_generator_valid, mask_generator_valid))

num_valid_samples = len(os.listdir('validation_3d/images/all_images'))

X_valid_seg = np.zeros((num_valid_samples, 256, 256, 1))
Y_valid_seg = np.zeros((num_valid_samples, 256, 256, 1))
for step in range(0, num_valid_samples):
    X_batch, Y_batch = next(valid_generator)
    X_valid_seg[step,:,:,:] = X_batch[0,:,:,:]
    Y_valid_seg[step,:,:,:] = np.round(Y_batch[0,:,:,:])

def generator(samples, batch_size = 16,  shuffle_data = True, resize = 256):
    while True:
        num_samples = len(samples)
        for offset in range(0, num_samples, batch_size):
            X, Y = next(train_generator)
            Y_yield = np.round(Y)
            yield X, Y_yield

image_files = glob.glob('train_3d/images/all_images/*')
train_generator_final = generator(image_files)

def dice_coeff(y_true, y_pred):
    _epsilon = 10 ** -7
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersections = K.sum(y_true_f * y_pred_f)
    unions = K.sum(y_true) + K.sum(y_pred)
    dice_scores = (2.0 * intersections + _epsilon) / (unions + _epsilon)
    return dice_scores

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss
  
get_custom_objects().update({"dice": dice_loss})

class LayerNormalization (Layer) :
    
    def call(self, x, mask=None, training=None) :
        axis = list (range (1, len (x.shape)))
        x /= K.std (x, axis = axis, keepdims = True) + K.epsilon()
        x -= K.mean (x, axis = axis, keepdims = True)
        return x
        
    def compute_output_shape(self, input_shape):
        return input_shape

def build_model(act_fn = 'relu', init_fn = 'he_normal', width=256, height = 256, channels = 1): 
    inputs = Input((width,height,channels))

    # note we use linear function before layer normalization
    conv1 = Conv2D(16, 5, activation = 'linear', padding = 'same', kernel_initializer = init_fn)(inputs)
    conv1 = LayerNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(32, 3, activation = act_fn, padding = 'same', kernel_initializer = init_fn)(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(64, 3, activation = 'linear', padding = 'same', kernel_initializer = init_fn)(pool2)
    conv3 = LayerNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(128, 3, activation = act_fn, padding = 'same', kernel_initializer = init_fn)(pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(144, 3, activation = act_fn, padding = 'same', kernel_initializer = init_fn)(pool4)

    up6 = Conv2D(128, 2, activation = 'linear', padding = 'same', kernel_initializer = init_fn)(UpSampling2D(size = (2,2))(conv5))
    up6 = LayerNormalization()(up6)
    merge6 = concatenate([conv4,up6], axis = 3)
    conv6 = Conv2D(128, 3, activation = act_fn, padding = 'same', kernel_initializer = init_fn)(merge6)

    up7 = Conv2D(64, 2, activation = act_fn, padding = 'same', kernel_initializer = init_fn)(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(64, 3, activation = act_fn, padding = 'same', kernel_initializer = init_fn)(merge7)

    up8 = Conv2D(32, 2, activation = 'linear', padding = 'same', kernel_initializer = init_fn)(UpSampling2D(size = (2,2))(conv7))
    up8 = LayerNormalization()(up8)
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(32, 3, activation = act_fn, padding = 'same', kernel_initializer = init_fn)(merge8)

    up9 = Conv2D(16, 2, activation = act_fn, padding = 'same', kernel_initializer = init_fn)(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(16, 3, activation = act_fn, padding = 'same', kernel_initializer = init_fn)(merge9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
    model = Model(inputs = inputs, outputs = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'dice', metrics=[dice_coeff])
    return model

num_epochs = 50

batch_size = 16

BACKBONE = 'inceptionv3'

base_model = sm.Unet(BACKBONE, encoder_weights='imagenet', classes=1, activation='sigmoid')
inp = Input(shape=(256, 256, 1))
l1 = Conv2D(3, (1, 1)) (inp)
out = base_model(l1)


model = Model(inp, out, name = base_model.name)
checkpointer = ModelCheckpoint('model_inception_3d.h5', verbose=1, save_best_only=True)

model.compile(optimizer = Adam(lr = 1e-4), loss=dice_loss, metrics=[dice_coeff])
steps_per_epoch = int(np.ceil(num_train_samples/batch_size))
#validation_steps = int(np.ceil(num_valid_samples/batch_size))

results = model.fit_generator(train_generator_final, validation_data = (X_valid_seg, Y_valid_seg), steps_per_epoch=steps_per_epoch, epochs = num_epochs, callbacks = [checkpointer], shuffle = False)

model.load_weights('model_inception_3d.h5')
preds_test = model.predict(X_valid_seg, verbose=1)
preds_test = (preds_test > 0.5).astype(np.uint8)

def np_dice(true, pred):
    intersection = np.sum(true * pred)
    dc =(2.0 * intersection) / (np.sum(true) + np.sum(pred))
    return dc

print("The dice score for this model is: ", np_dice(Y_valid_seg, preds_test))