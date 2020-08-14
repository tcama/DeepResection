import numpy as np
import glob
import os
import keras
from PIL import Image
import shutil

def getListOfImagePaths(dirPath):
    imagePaths = glob.glob('png_tle/*/img/*.png')
    ret = [os.path.join(dirPath, os.path.normpath(path)) for path in imagePaths]
    return ret

def getListOfMaskPaths(imagePaths):
    ret = [p.replace('img/', 'mask/') for p in imagePaths]
    return ret


dirPath = os.getcwd()
imagePaths = getListOfImagePaths(dirPath)
maskPaths = getListOfMaskPaths(imagePaths)

# define the training, validation, and test patients
imagePaths_valid = []
imagePaths_test = []
imagePaths_train = []
train_id = []
valid_id = []
test_id = []
for path in imagePaths:
    pat_id = (path.split('png_tle/')[1]).split('/img')[0]
    if 'HUP' in pat_id:
        pat_id = pat_id.split('_')[0]
    else:
        pat_id = pat_id.split('_dim')[0]
    if pat_id in ('14_w', '23_r', '27_m', 'pat05', 'pat11', 'pat15', 'pat20', 'HUP058', 'HUP056', 'HUP099', '127630'):
        imagePaths_valid.append(path)
        if pat_id not in valid_id:
            valid_id.append(pat_id)
    elif pat_id in ('24_c', '40_f', '42_m', 'pat03', 'pat06', 'pat25', 'pat30', 'HUP088', 'HUP094', 'HUP142', '127933'):
        imagePaths_test.append(path)
        if pat_id not in test_id:
            test_id.append(pat_id)
    else:
        imagePaths_train.append(path)
        if pat_id not in train_id:
            train_id.append(pat_id)
maskPaths_valid = getListOfMaskPaths(imagePaths_valid)
maskPaths_test = getListOfMaskPaths(imagePaths_test)
maskPaths_train = getListOfMaskPaths(imagePaths_train)

print(len(maskPaths_train))
print(len(maskPaths_valid))
print(len(maskPaths_test))

print('1')

for path in imagePaths_train:
    pat_id = (path.split('png_tle/')[1]).split('/img/')[0]
    new_name = os.path.basename(path)
    new_name = new_name.replace('img_', "%s_img_" % pat_id)
    new_file = "train/images/all_images/%s" % new_name
    shutil.move(path, new_file)

print('2')

for path in maskPaths_train:
    pat_id = (path.split('png_tle/')[1]).split('/mask/')[0]
    new_name = os.path.basename(path)
    new_name = new_name.replace('img_', "%s_img_" % pat_id)
    new_file = "train/masks/all_images/%s" % new_name
    shutil.move(path, new_file)

print('3')

for path in imagePaths_valid:
    pat_id = (path.split('png_tle/')[1]).split('/img/')[0]
    new_name = os.path.basename(path)
    new_name = new_name.replace('img_', "%s_img_" % pat_id)
    new_file = "validation/images/all_images/%s" % new_name
    shutil.move(path, new_file)

print('4')

for path in maskPaths_valid:
    pat_id = (path.split('png_tle/')[1]).split('/mask/')[0]
    new_name = os.path.basename(path)
    new_name = new_name.replace('img_', "%s_img_" % pat_id)
    new_file = "validation/masks/all_images/%s" % new_name
    shutil.move(path, new_file)

print('5')

for path in imagePaths_test:
    pat_id = (path.split('png_tle/')[1]).split('/img/')[0]
    new_name = os.path.basename(path)
    new_name = new_name.replace('img_', "%s_img_" % pat_id)
    new_file = "test/images/all_images/%s" % new_name
    shutil.move(path, new_file)

print('6')

for path in maskPaths_test:
    pat_id = (path.split('png_tle/')[1]).split('/mask/')[0]
    new_name = os.path.basename(path)
    new_name = new_name.replace('img_', "%s_img_" % pat_id)
    new_file = "test/masks/all_images/%s" % new_name
    shutil.move(path, new_file)
