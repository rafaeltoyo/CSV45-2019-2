import os
import cv2
import glob
import time
import keras
import datetime
import numpy as np
from keras.models import load_model

batch_size = 16
epochs = 1
saved_models_dir = 'saved_models'

def get_latest_model():
    return max(glob.glob(saved_models_dir+"/*"), key=os.path.getctime)

X_train = np.empty((0,256,256,3))
y_train = np.empty((0,256,256,1))
X_test = np.empty((0,256,256,3))
y_test = np.empty((0,256,256,1))

model_name = get_latest_model()
model = load_model(model_name)
print("Loaded model " + model_name)
#model.summary()

print("Loading training set..")
train_names = os.listdir(os.path.join('training','color'))[:1000]
for index, name in enumerate(train_names):
    # load data
    image = cv2.imread(os.path.join('training', 'color', name))
    image = cv2.resize(image, (256,256))
    mask = cv2.imread(os.path.join('training', 'mask', name), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (256,256)).reshape((256,256,1))
    X_train = np.append(X_train, [image], axis=0)
    y_train = np.append(y_train, [mask], axis=0)
    if (index % 500 == 0):
        print("%d of %d" % (index, len(train_names)))

print("Loading test set..")
test_names = os.listdir(os.path.join('evaluation','color'))[:100]
for index, name in enumerate(test_names):
    # load data
    image = cv2.imread(os.path.join('evaluation', 'color', name))
    image = cv2.resize(image, (256,256))
    mask = cv2.imread(os.path.join('evaluation', 'mask', name), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (256,256)).reshape((256,256,1))
    X_test = np.append(X_test, [image], axis=0)
    y_test = np.append(y_test, [mask], axis=0)
    if (index % 500 == 0):
        print("%d of %d" % (index, len(test_names)))

print("Starting to train..")
start_time = time.time()
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_test, y_test),
          shuffle=True)
print("Fitting duration: " + str(time.time() - start_time))

new_model_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.h5")
model.save(os.path.join(saved_models_dir, new_model_name))
print("Saved model " + new_model_name)

# Score trained model.
scores = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
