import numpy as np
import glob
import os
import keras
from PIL import image

def getListOfImagePaths(dirPath):
    imagePaths = glob.glob('data/png/*/img/*')
    ret = [os.path.join(dirPath, os.path.normpath(path)) for path in imagePaths]
    return ret

def getListOfMaskPaths(dirPath):
    maskPaths = glob.glob('data/png/*/mask/*')
    ret = [os.path.join(dirPath, os.path.normpath(path)) for path in maskPaths]
    return ret

# dim is a tuple: (192, 192)
def getImages(imagePaths, dim):
    ret = np.empty((dim, len(imagePaths)))
    index = 0
    for path in imagePaths:
        img = Image.open(path)
        img_arr = np.asarray(img)
        #crop image
        if(img_arr.shape != dim):
            if(img_arr.shape[0] != dim[0]):
                toBeRemoved = img_arr.shape[0]-dim[0]
                toBeRemoved1 = int(np.ceil(toBeRemoved/2))
                toBeRemoved2 = int(np.floor(toBeRemoved/2))
                index1 = toBeRemoved1+1
                index2 = img_arr.shape[0]-toBeRemoved2+1
                img_arr = img_arr[index1:index2,:]
            if(img_arr.shape[1] != dim[1]):
                toBeRemoved = img_arr.shape[1]-dim[1]
                toBeRemoved1 = int(np.ceil(toBeRemoved/2))
                toBeRemoved2 = int(np.floor(toBeRemoved/2))
                index1 = toBeRemoved1+1
                index2 = img_arr.shape[1]-toBeRemoved2+1
                img_arr = img_arr[:,index1:index2]
        #add image to returned array
        ret[:,:,index] = img_arr
        index=index+1
    return ret

# dim is a tuple: (192, 192)
def getMasks(imagePaths, dim):
    ret = np.empty((dim, len(imagePaths)))
    index = 0
    for path in imagePaths:
        img = Image.open(path)
        img_arr = np.asarray(img)
        #crop image
        if(img_arr.shape != dim):
            if(img_arr.shape[0] != dim[0]):
                toBeRemoved = img_arr.shape[0]-dim[0]
                toBeRemoved1 = int(np.ceil(toBeRemoved/2))
                toBeRemoved2 = int(np.floor(toBeRemoved/2))
                index1 = toBeRemoved1+1
                index2 = img_arr.shape[0]-toBeRemoved2+1
                img_arr = img_arr[index1:index2,:]
            if(img_arr.shape[1] != dim[1]):
                toBeRemoved = img_arr.shape[1]-dim[1]
                toBeRemoved1 = int(np.ceil(toBeRemoved/2))
                toBeRemoved2 = int(np.floor(toBeRemoved/2))
                index1 = toBeRemoved1+1
                index2 = img_arr.shape[1]-toBeRemoved2+1
                img_arr = img_arr[:,index1:index2]
        #add image to returned array
        ret[:,:,index] = img_arr
        index=index+1
    return ret

class DataGenerator:
    def __init__(self, dirPath, batchSize=32, dim=(192, 192)):
        self.dim = dim
        self.imagePaths = getListOfImagePaths(dirPath)
        self.maskPaths = getListOfMaskPaths(dirPath)
        self.allImages = getImages(self.imagePaths, dim)
        self.allMasks = getMasks(self.maskPaths, dim)
        self.batchSize = batchSize
        self.numBatches = np.ceil(len(self.imagePaths)/self.batchSize)
    
    # i is the ith batch, starting at 0
    def generateBatch(i):
        index1 = i*self.batchSize
        index2 = np.minimum(index1+self.batchSize, len(self.imagePaths))
        batchImages = self.allImages[:,:,index1:index2]
        batchMasks = self.allMasks[:,:,index1:index2]
        return batchImages, batchMasks