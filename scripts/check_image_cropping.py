import numpy as np
import glob
import os
from PIL import Image

def getListOfImagePaths(dirPath):
    imagePaths = glob.glob('data/png/*/img/*')
    ret = [os.path.join(dirPath, os.path.normpath(path)) for path in imagePaths]
    return ret

def getListOfMaskPaths(dirPath):
    maskPaths = glob.glob('data/png/*/mask/*')
    ret = [os.path.join(dirPath, os.path.normpath(path)) for path in maskPaths]
    return ret

# dim is a tuple: (192, 192)
def checkImages(imagePaths, dim=(192, 192)):
    ret = np.empty((len(imagePaths)))
    index = 0
    for path in imagePaths:
        img = Image.open(path)
        img_arr = np.asarray(img)
        #crop image
        bool1 = False
        bool2 = False
        if(img_arr.shape != dim):
            if(img_arr.shape[0] != dim[0]):
                toBeRemoved = img_arr.shape[0]-dim[0]
                toBeRemoved1 = int(np.ceil(toBeRemoved/2))
                toBeRemoved2 = int(np.floor(toBeRemoved/2))
                index1 = toBeRemoved1
                index2 = img_arr.shape[1]-toBeRemoved2
                removed1 = img_arr[0:index1,:]
                removed2 = img_arr[index2:,:]
                if(np.any(removed1) | np.any(removed2)):
                    bool1 = True
            if(img_arr.shape[1] != dim[1]):
                toBeRemoved = img_arr.shape[1]-dim[1]
                toBeRemoved1 = int(np.ceil(toBeRemoved/2))
                toBeRemoved2 = int(np.floor(toBeRemoved/2))
                index1 = toBeRemoved1
                index2 = img_arr.shape[1]-toBeRemoved2
                removed1 = img_arr[:,0:index1]
                removed2 = img_arr[:,index2:]
                if(np.any(removed1) | np.any(removed2)):
                    bool2 = True
        #add image to returned array
        ret[index] = bool1 | bool2
        index=index+1
    return ret

# dim is a tuple: (192, 192)
def checkMasks(maskPaths, dim=(192, 192)):
    ret = np.empty((len(imagePaths)))
    index = 0
    for path in maskPaths:
        img = Image.open(path)
        img_arr = np.asarray(img)
        #crop image
        bool1 = False
        bool2 = False
        if(img_arr.shape != dim):
            if(img_arr.shape[0] != dim[0]):
                toBeRemoved = img_arr.shape[0]-dim[0]
                toBeRemoved1 = int(np.ceil(toBeRemoved/2))
                toBeRemoved2 = int(np.floor(toBeRemoved/2))
                index1 = toBeRemoved1
                index2 = img_arr.shape[1]-toBeRemoved2
                removed1 = img_arr[0:index1,:]
                removed2 = img_arr[index2:,:]
                if(np.any(removed1) | np.any(removed2)):
                    bool1 = True
            if(img_arr.shape[1] != dim[1]):
                toBeRemoved = img_arr.shape[1]-dim[1]
                toBeRemoved1 = int(np.ceil(toBeRemoved/2))
                toBeRemoved2 = int(np.floor(toBeRemoved/2))
                index1 = toBeRemoved1
                index2 = img_arr.shape[1]-toBeRemoved2
                removed1 = img_arr[:,0:index1]
                removed2 = img_arr[:,index2:]
                if(np.any(removed1) | np.any(removed2)):
                    bool2 = True
        #add image to returned array
        ret[index] = bool1 | bool2
        index=index+1
    return ret


# script
dirPath = os.getcwd()

imagePaths = getListOfImagePaths(dirPath)
ret_img = checkImages(imagePaths)

maskPaths = getListOfMaskPaths(dirPath)
ret_mask = checkMasks(maskPaths)
    