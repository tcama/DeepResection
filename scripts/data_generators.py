import numpy as np
import glob
import os
import keras
from PIL import Image
from sklearn.model_selection import train_test_split

def getListOfImagePaths(dirPath):
    imagePaths = glob.glob('data/png/*/img/*')
    ret = [os.path.join(dirPath, os.path.normpath(path)) for path in imagePaths]
    return ret

def getListOfMaskPaths(imagePaths):
    ret = [p.replace('img/', 'mask/') for p in imagePaths]
    return ret

# dim is a tuple: (192, 192)
def getImagesForClassification(imagePaths, dim=(192, 192)):
    ret = np.empty((len(imagePaths), dim[0], dim[1], 3))
    index = 0
    for path in imagePaths:
        img = Image.open(path)
        img_arr = np.asarray(img)
        #crop image
        if(img_arr.shape[0] != dim[0]):
            toBeRemoved = img_arr.shape[0]-dim[0]
            toBeRemoved1 = int(np.ceil(toBeRemoved/2))
            toBeRemoved2 = int(np.floor(toBeRemoved/2))
            index1 = toBeRemoved1
            index2 = img_arr.shape[0]-toBeRemoved2
            img_arr = img_arr[index1:index2,:]
        if(img_arr.shape[1] != dim[1]):
            toBeRemoved = img_arr.shape[1]-dim[1]
            toBeRemoved1 = int(np.ceil(toBeRemoved/2))
            toBeRemoved2 = int(np.floor(toBeRemoved/2))
            index1 = toBeRemoved1
            index2 = img_arr.shape[1]-toBeRemoved2
            img_arr = img_arr[:,index1:index2]
        #add image to returned array
        ret[index,:,:,0] = img_arr
        ret[index,:,:,1] = img_arr
        ret[index,:,:,2] = img_arr
        index=index+1
    return ret

# returns numpy vector where each entry is 0 or 1 based on whether the image contains resected tissue
def getClassificationY(maskPaths):
    num_masks = len(maskPaths)
    Y = np.zeros(num_masks)
    index = 0
    for path in imagePaths:
        img = Image.open(path)
        img_arr = np.asarray(img)
        #crop image
        if(img_arr.shape[0] != dim[0]):
            toBeRemoved = img_arr.shape[0]-dim[0]
            toBeRemoved1 = int(np.ceil(toBeRemoved/2))
            toBeRemoved2 = int(np.floor(toBeRemoved/2))
            index1 = toBeRemoved1
            index2 = img_arr.shape[0]-toBeRemoved2
            img_arr = img_arr[index1:index2,:]
        if(img_arr.shape[1] != dim[1]):
            toBeRemoved = img_arr.shape[1]-dim[1]
            toBeRemoved1 = int(np.ceil(toBeRemoved/2))
            toBeRemoved2 = int(np.floor(toBeRemoved/2))
            index1 = toBeRemoved1
            index2 = img_arr.shape[1]-toBeRemoved2
            img_arr = img_arr[:,index1:index2]
        Y[index] = 1 if np.any(img_arr) else 0
        index=index+1
    return Y

# dim is a tuple: (256, 256)
def getImagesForSegmentation(imagePaths, dim=(256,256)):
    ret = np.empty((len(imagePaths), dim[0], dim[1]))
    index = 0
    for path in imagePaths:
        img = Image.open(path)
        img_arr = np.asarray(img)
        #pad image
        if(img_arr.shape[0] != dim[0]):
            toBeAdded = dim[0] - img_arr.shape[0]
            toBeAdded1 = int(np.ceil(toBeAdded/2))
            toBeAdded2 = int(np.floor(toBeAdded/2))
            img_arr = np.pad(img_arr, ((toBeAdded1, toBeAdded2), (0, 0)))
        if(img_arr.shape[1] != dim[1]):
            toBeAdded = dim[1] - img_arr.shape[1]
            toBeAdded1 = int(np.ceil(toBeAdded/2))
            toBeAdded2 = int(np.floor(toBeAdded/2))
            img_arr = np.pad(img_arr, ((0, 0), (toBeAdded1, toBeAdded2)))
        #add image to returned array
        ret[index,:,:] = img_arr
        index=index+1
    return ret


dirPath = os.getcwd()
imagePaths = getListOfImagePaths(dirPath)
maskPaths = getListOfMaskPaths(imagePaths)

# define the training, validation, and test patients
imagePaths_valid = []
for path in imagePaths:
    if '14_w' in path or '23_r' in path or '27_m' in path or 'pat05' in path or 'pat11' in path or 'pat15' in path or 'pat20' in path:
        imagePaths_valid.append(path)
maskPaths_valid = getListOfMaskPaths(imagePaths_valid)
imagePaths_test = []
for path in imagePaths:
    if '24_c' in path or '40_f' in path or '42_m' in path or 'pat03' in path or 'pat06' in path or 'pat25' in path or 'pat30' in path:
        imagePaths_test.append(path)
maskPaths_test = getListOfMaskPaths(imagePaths_test)
imagePaths_train = []
for path in imagePaths:
    if path not in imagePaths_valid and path not in imagePaths_test:
        imagePaths_train.append(path)
maskPaths_train = getListOfMaskPaths(imagePaths_train)

# get X and Y for each set, for the classification of images containing and not containing resected tissue
X_train_class = getImagesForClassification(imagePaths_train)
X_train_class = X_train_class/255
X_valid_class = getImagesForClassification(imagePaths_valid)
X_valid_class = X_valid_class/255
X_test_class = getImagesForClassification(imagePaths_test)
X_test_class = X_test_class/255
Y_train_class = getClassificationY(maskPaths_train)
Y_valid_class = getClassificationY(maskPaths_valid)
Y_test_class = getClassificationY(maskPaths_test)

# get X and Y for each set, for the segmentation of resected tissue in images
X_train_seg = getImagesForSegmentation(imagePaths_train)
X_train_seg = X_train_seg/255
X_train_seg = X_train_seg[..., np.newaxis]
X_valid_seg = getImagesForSegmentation(imagePaths_valid)
X_valid_seg = X_valid_seg/255
X_valid_seg = X_valid_seg[..., np.newaxis]
X_test_seg = getImagesForSegmentation(imagePaths_test)
X_test_seg = X_test_seg/255
X_test_seg = X_test_seg[..., np.newaxis]
Y_train_seg = getImagesForSegmentation(maskPaths_train)
Y_train_seg = Y_train_seg/255
Y_train_seg = (Y_train_seg > 0).astype(np.float32)
Y_train_seg = Y_train_seg[..., np.newaxis]
Y_valid_seg = getImagesForSegmentation(maskPaths_valid)
Y_valid_seg = Y_valid_seg/255
Y_valid_seg = (Y_valid_seg > 0).astype(np.float32)
Y_valid_seg = Y_valid_seg[..., np.newaxis]
Y_test_seg = getImagesForSegmentation(maskPaths_test)
Y_test_seg = Y_test_seg/255
Y_test_seg = (Y_test_seg > 0).astype(np.float32)
Y_test_seg = Y_test_seg[..., np.newaxis]

# save preprocessed dataset to npy files
np.save('X_train_class.npy', X_train_class)
np.save('Y_train_class.npy', Y_train_class)
np.save('X_valid_class.npy', X_valid_class)
np.save('Y_valid_class.npy', Y_valid_class)
np.save('X_test_class.npy', X_test_class)
np.save('Y_test_class.npy', Y_test_class)
np.save('X_train_seg.npy', X_train_seg)
np.save('Y_train_seg.npy', Y_train_seg)
np.save('X_valid_seg.npy', X_valid_seg)
np.save('Y_valid_seg.npy', Y_valid_seg)
np.save('X_test_seg.npy', X_test_seg)
np.save('Y_test_seg.npy', Y_test_seg)