
import os
import dlib
import cv2
import numpy as np
import urllib.request as ur
import tensorflow as tf

from PIL import Image

from keras.layers import Dense
from keras.models import Sequential
from keras.applications.resnet50 import ResNet50

# Needs to be here if using NVIDIA GPU, otherwise model wouldnt fit
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

CURRENT_PATH = os.path.split(os.path.realpath(__file__))[0]
PARENT_PATH = os.path.dirname(os.path.dirname(CURRENT_PATH))
MODEL_PATH = os.path.join(PARENT_PATH, "common",
                          "mmod_human_face_detector.dat")

TRAINEDMODEL_FILE = os.path.join(PARENT_PATH, "common", "model-ldl-resnet.h5")

cnn_face_detector = dlib.cnn_face_detection_model_v1(MODEL_PATH)

# Load our trained NN
resnet = ResNet50(include_top=False, pooling='avg')
model = Sequential()
model.add(resnet)
model.add(Dense(5, activation='softmax'))
model.layers[0].trainable = False

model.load_weights(TRAINEDMODEL_FILE)


def convert_url_to_imgs(urls):
    imgs = []
    for u in urls:
        if len(u) == 0:
            continue

        resp = ur.urlopen(u)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        imgs.append(image)
    return imgs


def score_mapping(modelScore):

    if modelScore <= 1.9:
        mappingScore = ((4 - 2.5) / (1.9 - 1.0)) * (modelScore-1.0) + 2.5
    elif modelScore <= 2.8:
        mappingScore = ((5.5 - 4) / (2.8 - 1.9)) * (modelScore-1.9) + 4
    elif modelScore <= 3.4:
        mappingScore = ((6.5 - 5.5) / (3.4 - 2.8)) * (modelScore-2.8) + 5.5
    elif modelScore <= 4:
        mappingScore = ((8 - 6.5) / (4 - 3.4)) * (modelScore-3.4) + 6.5
    elif modelScore < 5:
        mappingScore = ((9 - 8) / (5 - 4)) * (modelScore-4) + 8

    return mappingScore


def show_output(img, face, score):
    cv2.rectangle(img, (face[0], face[1]),
                  (face[2], face[3]), (0, 255, 0), 3)
    cv2.putText(img, str('%.2f' % (score)), (face[0], face[3]), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2)

    im_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    im_pil.show()


def beauty_predict(img):

    # Get faces
    faces = cnn_face_detector(img, 0)
    
    try:
        for i, f in enumerate(faces):
            face = [f.rect.left(), f.rect.top(), f.rect.right(), f.rect.bottom()]
            croppedImg = img[face[1]: face[3], face[0]: face[2], :]

            # Resized img and normalized it
            resizedImg = cv2.resize(croppedImg, (224, 244))
            normImg = np.array([(resizedImg - 127.5) / 127.5])

            # Get prediction from NN
            prediction = model.predict(normImg)

            # Normalized prediction
            scList = prediction[0]
            score = 1 * scList[0] + 2 * scList[1] + 3 * \
                scList[2] + 4 * scList[3] + 5 * scList[4]

            print("Score: " + str(score))

            # Show result
            # show_output(img, face, score)

            # We only care for one face
            return score
    except Exception as e:
        print(str(e))
    return 0
