import data

# System lib
import os

# Libraries for manipulating Dataset
import cv2
import numpy as np

import urllib.request as ur

CURRENT_PATH = os.path.split(os.path.realpath(__file__))[0]
PARENT_PATH = os.path.dirname(CURRENT_PATH)

MODEL_PATH = os.path.join(PARENT_PATH, "common",
                          "haarcascade_frontalface_alt.xml")

face_cascade = cv2.CascadeClassifier(MODEL_PATH)

def getFace(detector, img):

    # Convert img to grayscale to remove colour skin discrimination
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    w = img.shape[1]
    faces = detector.detectMultiScale(gray, 1.1, 5, 0, (w//2, w//2))

    resized_img = 0

    # Discard imgs with several faces
    if len(faces) == 1:
        face = faces[0]
        croped_img = img[face[1]:face[1]+face[3], face[0]:face[0]+face[2], :]
        resized_img = cv2.resize(croped_img, (224, 224))

        if resized_img.shape[0] != 224 or resized_img.shape[1] != 224:
            print("Invalid WxH")
    
    return resized_img

def convert_url_to_img(u):
    try:
        resp = ur.urlopen(u)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image
    except:
        print("error")

idx = 0
idy = 0
for u in data.data:
    print("Reading img: " + str(idy) + " of " + str(len(data.data)))
    idy += 1
    img = convert_url_to_img(u)
    if isinstance(img, np.ndarray):
        img = getFace(face_cascade, img)
        if isinstance(img, np.ndarray):
            result = "./Images/{}.jpg".format(idx)
            idx += 1
            cv2.imwrite(result, img)