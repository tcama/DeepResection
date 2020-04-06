import numpy as np
import glob
import os
import keras
from PIL import image
from keras.applications.densenet import DenseNet121
from sklearn.model_selection import train_test_split

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
                index1 = toBeRemoved1
                index2 = img_arr.shape[1]-toBeRemoved2
                img_arr = img_arr[index1:index2,:]
            if(img_arr.shape[1] != dim[1]):
                toBeRemoved = img_arr.shape[1]-dim[1]
                toBeRemoved1 = int(np.ceil(toBeRemoved/2))
                toBeRemoved2 = int(np.floor(toBeRemoved/2))
                index1 = toBeRemoved1
                index2 = img_arr.shape[1]-toBeRemoved2
                img_arr = img_arr[:,index1:index2]
        #add image to returned array
        ret[:,:,index] = img_arr
        index=index+1
    return ret

# dim is a tuple: (192, 192)
def getMasks(maskPaths, dim):
    ret = np.empty((dim, len(imagePaths)))
    index = 0
    for path in maskPaths:
        img = Image.open(path)
        img_arr = np.asarray(img)
        #crop image
        if(img_arr.shape != dim):
            if(img_arr.shape[0] != dim[0]):
                toBeRemoved = img_arr.shape[0]-dim[0]
                toBeRemoved1 = int(np.ceil(toBeRemoved/2))
                toBeRemoved2 = int(np.floor(toBeRemoved/2))
                index1 = toBeRemoved1
                index2 = img_arr.shape[1]-toBeRemoved2
                img_arr = img_arr[index1:index2,:]
            if(img_arr.shape[1] != dim[1]):
                toBeRemoved = img_arr.shape[1]-dim[1]
                toBeRemoved1 = int(np.ceil(toBeRemoved/2))
                toBeRemoved2 = int(np.floor(toBeRemoved/2))
                index1 = toBeRemoved1
                index2 = img_arr.shape[1]-toBeRemoved2
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
        self.currentBatchNum = 0
        X = self.allImages
        Y = np.empty(len(self.maskPaths))
        for maskInd in range(0, len(self.maskPaths)):
            Y[maskInd] = 1 if np.any(self.allMasks[:,:,maskInd]) else 0
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size = 0.25)
        self.remainingIndices = np.random.shuffle(np.arange(self.Y_train.size))
        self.numBatches = np.ceil(self.Y_train.size/self.batchSize)
    
    # i is the ith batch, starting at 0
    def generateBatchForClassification():
        lastInd = np.minimum(self.remainingIndices.size, self.batchSize)
        indices = self.remainingIndices[0:lastInd]
        batchX = self.X_train[:,:,indices]
        batchY = self.Y_train[indices]
        self.currentBatchNum=self.currentBatchNum+1
        self.remainingIndices = np.delete(self.remainingIndices, indices)
        if self.remainingIndices.size == 0:
            self.on_epoch_end()
        return batchX, batchY
    
    def on_epoch_end():
        self.remainingIndices = np.random.shuffle(np.arange(self.Y_train.size))
    
    def getNumBatches():
        return self.numBatches

    def getTrainDataForClassification():
        old_shape = self.X_train.shape
        new_shape = (old_shape[0], old_shape[1], 3, old_shape[2])
        new_X_train = np.empty(new_shape)
        for i in range(0, old_shape[2]):
            new_X_train[:,:,0,i] = self.X_train[:,:,i]
            new_X_train[:,:,1,i] = self.X_train[:,:,i]
            new_X_train[:,:,2,i] = self.X_train[:,:,i]
        return new_X_train, self.Y_train
    
    def getTestDataForClassification():
        old_shape = self.X_test.shape
        new_shape = (old_shape[0], old_shape[1], 3, old_shape[2])
        new_X_test = np.empty(new_shape)
        for i in range(0, old_shape[2]):
            new_X_test[:,:,0,i] = self.X_test[:,:,i]
            new_X_test[:,:,1,i] = self.X_test[:,:,i]
            new_X_test[:,:,2,i] = self.X_test[:,:,i]
        return new_X_test, self.Y_test
