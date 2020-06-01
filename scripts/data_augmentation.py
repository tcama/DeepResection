import numpy as np
from tensorflow.keras.preprocessing.image import random_brightness
from keras.preprocessing.image import ImageDataGenerator

# returns the original dataset, except with random brightness adjustments for data augmentation
def brightness_augmentation(X_train):
    (num_train_samples, _, _, _) = X_train.shape
    X_train_b = np.zeros(X_train.shape)
    for i in range(0, num_train_samples):
        img = X_train[i,:,:,:]
        img_b = random_brightness(img, brightness_range=(0.7, 1.3))
        X_train_b[i,:,:,:] = img_b
    X_train_b = X_train_b/255
    return X_train_b

# load dataset
X_train_class = np.load('X_train_class.npy')
Y_train_class = np.load('Y_train_class.npy')
X_valid_class = np.load('X_valid_class.npy')
Y_valid_class = np.load('Y_valid_class.npy')
X_test_class = np.load('X_test_class.npy')
Y_test_class = np.load('Y_test_class.npy')

X_train_seg = np.load('X_train_seg.npy')
Y_train_seg = np.load('Y_train_seg.npy')
X_valid_seg = np.load('X_valid_seg.npy')
Y_valid_seg = np.load('Y_valid_seg.npy')
X_test_seg = np.load('X_test_seg.npy')
Y_test_seg = np.load('Y_test_seg.npy')

# define arguments for further data augmentation (translation, rotation, reflection)
seed = 909
datagen_args = dict(rotation_range = 10, width_shift_range = 0.1, height_shift_range = 0.1, horizontal_flip = True, vertical_flip = True)
image_datagen_class = ImageDataGenerator(**datagen_args)
image_datagen_seg = ImageDataGenerator(**datagen_args)
mask_datagen = ImageDataGenerator(**datagen_args)

X_train_class_b = brightness_augmentation(X_train_class)
X_train_seg_b = brightness_augmentation(X_train_seg)

# fit the data augmentation generators to the trainin sets for both the classification and segmentation problems
image_datagen_class.fit(X_train_class_b, Y_train_class, augment = True, seed = seed)

image_datagen_seg.fit(X_train_seg_b, augment = True, seed = seed)
mask_datagen.fit(Y_train_seg, augment = True, seed = seed)

train_generator_class = image_datagen_class.flow(X_train_class_b, Y_train_class, seed = seed, batch_size = 32)

image_generator_seg = image_datagen_seg.flow(X_train_seg_b, seed = seed, batch_size=1)
mask_generator = mask_datagen.flow(Y_train_seg, seed = seed, batch_size=1)
train_generator_seg = (pair for pair in zip(image_generator_seg, mask_generator))

(num_train_samples, _, _, _) = X_train_seg.shape

# for the segmentation problem, we need to get back the train data
X_train_seg_f = np.zeros(X_train_seg.shape)
Y_train_seg_f = np.zeros(Y_train_seg.shape)

for step in range(0, num_train_samples):
    X_batch, Y_batch = next(train_generator)
    X_train_seg_f[step,:,:,:] = X_batch[0,:,:,:]
    Y_train_seg_f[step,:,:,:] = np.round(Y_batch[0,:,:,:])