import numpy as np
import glob
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.applications.densenet import DenseNet121
from PIL import image

def generate_classification_model(dim=(192, 192)):
    #model = Sequential()
    #model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(dim[0], dim[1], 1)))
    #model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(dim[0], dim[1], 1)))
    #model.add(MaxPooling2D(poolsize=(2,2)))
    #model.add(Dropout=0.25)

    #model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(dim[0], dim[1], 1)))
    #model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(dim[0], dim[1], 1)))
    #model.add(MaxPooling2D(poolsize=(2,2)))
    #model.add(Dropout=0.25)

    #model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(dim[0], dim[1], 1)))
    #model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(dim[0], dim[1], 1)))
    #model.add(MaxPooling2D(poolsize=(2,2)))
    #model.add(Dropout=0.25)

    #model.add(Flatten())
    #model.add(Dense(256, activation='relu'))
    #model.add(Dropout=0.25)
    #model.add(Dense(2, activation='softmax'))

    # use pre-trained weights on ImageNet
    model = DenseNet121(include_top=False, weights='imagenet', input_shape=(192,192, 3), pooling=max, classes=2)  

    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adm = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=adm, metrics=['accuracy'])
    return model
