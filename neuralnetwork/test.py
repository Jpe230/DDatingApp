import tensorflow as tf
import pickle
import numpy as np
import cv2
import os

from keras.applications import resnet50
from keras.models import Sequential
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout

# User-defined const
import const

# Needs to be here if using NVIDIA GPU, otherwise model wouldnt load
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Load ResNet50 as our base
resnet = ResNet50(include_top=False, pooling='avg')
model = Sequential()
model.add(resnet)
model.add(Dense(5, activation='softmax'))
model.layers[0].trainable = False

# Load our trained weights
model.load_weights(const.TRAINEDMODEL_FILE)

image_data = []
attract_data = []
PC = []

score = []
predScore = []


def detectFace(detector, image_path, image_name):
    imgAbsPath = image_path + image_name
    img = cv2.imread(imgAbsPath)
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    w = img.shape[1]
    faces = detector.detectMultiScale(gray, 1.1, 5, 0, (w//2, w//2))

    resized_im = 0

    if len(faces) == 1:
        face = faces[0]
        croped_im = img[face[1]:face[1]+face[3], face[0]:face[0]+face[2], :]
        resized_im = cv2.resize(croped_im, (224, 224))
    else:
        print(image_name+" error " + str(len(faces)))
    return resized_im

# Load CV2 face detector
face_cascade = cv2.CascadeClassifier(const.MODEL_PATH)

# Load test data
test_data = pickle.load(open(const.TESTING_FILE, 'rb'))
data_len = test_data.__len__()
test_label_dist = train_Y = [
    x for x in test_data[0:data_len]]

# Test the data
for i in range(0, data_len):

    label_distribution = test_label_dist[i]
    image = label_distribution[1]

    print("dist:" + str(label_distribution[0]))
    label_score = 1*label_distribution[2][0] + 2*label_distribution[2][1] + 3 * \
        label_distribution[2][2] + 4 * \
        label_distribution[2][3] + 5*label_distribution[2][4]
    print("score:%1.2f " % (label_score))
    score.append(label_score)

    # Predict with our model
    pred = model.predict(np.expand_dims(image, axis=0))

    ldList = pred[0]
    pred = 1 * ldList[0] + 2 * ldList[1] + 3 * \
        ldList[2] + 4 * ldList[3] + 5 * ldList[4]
    print("prediction:" + str(pred))
    predScore.append(pred)

# Get Correletion
y = np.asarray(score)
pred_y = np.asarray(predScore)
corr = np.corrcoef(y, pred_y)[0, 1]

# 1          = PERFECT
# 0.50 - .99 = High
# 0.30 - .49 = Moderate
# 0.00 - .29 = Low
# 0          = No correlation

# Tested correlation was around .0.46
print('PC (Pearson correlation) mean = %1.2f ' % (corr))
