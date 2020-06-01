from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy.ma as ma
import segmentation_models as sm
from keras.callbacks import ModelCheckpoint

# RUN THE CLASSIFICATION MODEL (run from DeepResection directory)
model_detect_resection = generate_classification_model()

num_epochs = 10

batch_size = 32

steps_per_epoch = np.ceil(num_train_samples/32.0)

model_detect_resection.fit_generator(train_generator_class, steps_per_epoch = steps_per_epoch, epochs=num_epochs)
score=model_detect_resection.evaluate(X_test_class, Y_test_class, batch_size=batch_size)

# CALCULATE AND PLOT THE ROC CURVE FOR THE CLASSIFICATION PROBLEM
Y_score = model_detect_resection.predict(X_test)
fpr, tpr, thresholds = roc_curve(Y_test, Y_score)
auc = roc_auc_score(Y_test, Y_score)
print('auc: ')
print(auc)

plt.plot(fpr, tpr)
plt.axis([-0.1, 1, 0, 1.1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve: Classification of Resection Zone Images')
plt.show()

# VISUALIZE AUGMENTED TRAINING DATA FOR SEGMENTATION PROBLEM
fig=plt.figure(figsize=(130, 130), dpi = 75)

i=0
while np.sum(Y_train_seg_f[i,:,:,0]) < 200:
    i = i+1

X = X_train_seg_f[i,:,:,0]
Y = Y_train_seg_f[i,:,:,0]

mask = ma.masked_where(Y != 1.0, Y)
    
fig.add_subplot(3, 3, 1)
plt.imshow(X, cmap = "gray")
plt.title("Image")
plt.axis('off')

fig.add_subplot(3, 3, 2)
plt.imshow(X, cmap = "gray")
plt.imshow(mask, 'cool', alpha=0.7)
plt.title("Image with mask")
plt.axis('off')

fig.add_subplot(3, 3, 3)
plt.imshow(Y, 'cool', alpha=0.7)
plt.title("Mask")
plt.axis('off')
    
plt.subplots_adjust(bottom=0.1, left = 0.01, right=0.05, top=0.3)
plt.show()

# TRAIN THE MAGICIANS CORNER MODEL
num_epochs = 35

batch_size = 64

model = build_model(act_fn = 'relu', init_fn = 'he_normal', width=256, height = 256, channels = 1)
checkpointer = ModelCheckpoint('model_mc.h5', verbose=1, save_best_only=True)

model.fit(X_train_seg_f, Y_train_seg_f, validation_data = (X_valid_seg, Y_valid_seg), batch_size = batch_size, epochs = num_epochs, callbacks = [checkpointer])

# VISUALIZE MODEL PREDICTIONS ON TEST DATA AND COMPARE TO GROUND TRUTH
model.load_weights('./model_mc.h5')
preds_test = model.predict(X_test_seg, verbose=1)
preds_test = (preds_test > 0.5).astype(np.uint8)

def np_dice(true, pred):
    intersection = np.sum(true * pred)
    dc =(2.0 * intersection) / (np.sum(true) + np.sum(pred))
    return dc

fig=plt.figure(figsize=(130, 130), dpi = 75)

i=0
for j in range(0,8,2):
    while np.sum(Y_test_seg[i,:,:,0]) < 50:
        i = i+1
        print()
    image = X_test_seg[i,...,0]
    mask =  Y_test_seg[i,...,0]
    mask = ma.masked_where(mask == 0, mask)
    pred = preds_test[i,...,0]
    pred = ma.masked_where(pred == 0, pred)
    
    fig.add_subplot(8, 2, j+1)
    plt.imshow(image, cmap = "gray")
    plt.imshow(mask, 'cool', alpha=0.7)
    plt.title("Ground Truth")
    plt.axis('off')
    
    fig.add_subplot(8, 2, j+2)
    plt.imshow(image, cmap = "gray")
    plt.imshow(pred, 'cool', alpha=0.7)
    plt.title("Prediction")
    plt.axis('off')
    i = i+20
    
plt.subplots_adjust(bottom=0.1, left = 0.01, right=0.05, top=0.3)
plt.show()

print("The dice score for this model is: ", np_dice(Y_test_seg, preds_test))

# TRAIN A NEW MODEL FROM THE "SEGMENTATION MODELS" GITHUB CODEBASE, THIS MODEL HAS AN ENCODER BACKBONE WITH A SPECIFIED ARCHITECTURE
# AND PRETRAINED WEIGHTS FROM THE IMAGENET DATASET

# change the value of "backbone" based on what kind of encoder architecture you want to use
BACKBONE = 'vgg16'

base_model = sm.Unet(BACKBONE, encoder_weights='imagenet', classes=1, activation='sigmoid')
inp = Input(shape=(256, 256, 1))
l1 = Conv2D(3, (1, 1)) (inp)
out = base_model(l1)
model = Model(inp, out, name = base_model.name)
checkpointer = ModelCheckpoint('model_vgg16.h5', verbose=1, save_best_only=True)
model.compile(optimizer = Adam(lr = 1e-4), loss=dice_loss, metrics=[dice_coeff])
steps_per_epoch = np.ceil(num_train_samples/64.0)
batch_size = 16
results = model.fit(X_train_seg_f, Y_train_seg_f, validation_data = (X_valid_seg, Y_valid_seg), batch_size=batch_size, epochs = num_epochs, callbacks=[checkpointer])

# AGAIN VISUALIZE MODEL PREDICTIONS ON TEST DATA AND COMPARE TO GROUND TRUTH
model.load_weights('./model_vgg16.h5')
preds_test = model.predict(X_test_seg, verbose=1)
preds_test = (preds_test > 0.5).astype(np.uint8)

def np_dice(true, pred):
    intersection = np.sum(true * pred)
    dc =(2.0 * intersection) / (np.sum(true) + np.sum(pred))
    return dc

fig=plt.figure(figsize=(130, 130), dpi = 75)

i=0
for j in range(0,8,2):
    while np.sum(Y_test_seg[i,:,:,0]) < 50:
        i = i+1
        print()
    image = X_test_seg[i,...,0]
    mask =  Y_test_seg[i,...,0]
    mask = ma.masked_where(mask == 0, mask)
    pred = preds_test[i,...,0]
    pred = ma.masked_where(pred == 0, pred)
    
    fig.add_subplot(8, 2, j+1)
    plt.imshow(image, cmap = "gray")
    plt.imshow(mask, 'cool', alpha=0.7)
    plt.title("Ground Truth")
    plt.axis('off')
    
    fig.add_subplot(8, 2, j+2)
    plt.imshow(image, cmap = "gray")
    plt.imshow(pred, 'cool', alpha=0.7)
    plt.title("Prediction")
    plt.axis('off')
    i = i+20
    
plt.subplots_adjust(bottom=0.1, left = 0.01, right=0.05, top=0.3)
plt.show()

print("The dice score for this model is: ", np_dice(Y_test_seg, preds_test))