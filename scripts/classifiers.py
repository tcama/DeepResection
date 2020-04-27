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
    def generate_classification_model():

    # use pre-trained weights on ImageNet as starting weights
    model = DenseNet121(include_top=False, weights='imagenet', input_shape=(192,192,3), pooling=max)

    # add sigmoid activation layer
    top_model = GlobalAveragePooling2D(input_shape=(192,192,3)) (model.layers[-1].output)
    top_model2 = Dense(1, activation='sigmoid') (top_model)
    #model.add(layers.GlobalAveragePooling2D(input_shape=(192,192,3)))
    #model.add(layers.Dense(1, activation='sigmoid'))
    #model.summary()

    #model.layers.pop()

    new_model = Model(inputs=model.input, outputs=top_model2)

    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adm = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    new_model.compile(loss='binary_crossentropy', optimizer=adm, metrics=['binary_accuracy'])
    return new_model