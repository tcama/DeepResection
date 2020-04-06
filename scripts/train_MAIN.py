import numpy as np
import glob
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
import data_generators as dg
import classifiers as cl
from PIL import image

# run from DeepResection directory
current_dir = os.getcwd()

data_generator = dg.DataGenerator(current_dir)

model_detect_resection = cl.generate_classification_model()

num_epochs = 10

batch_size = 32

X_train, Y_train = dg.getTrainDataForClassification()
X_test, Y_test = dg.getTestDataForClassification()

model_detect_resection.fit(X_train, Y_train, batch_size=batch_size, epochs=num_epochs)
score=model_detect_resection.evaluate(X_test, Y_test, batch_size=batch_size)

print('metrics: ')
print(model_detect_resection.metrics_names)
print('scores:')
print(score)