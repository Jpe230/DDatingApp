# System lib
import os

# Libraries for manipulating Dataset
import cv2
import pickle
import numpy as np
import numpy
from PIL import ImageEnhance

# Libraries for downloading Dataset
import zipfile
import gdown
import random

from numpy.core.fromnumeric import resize

# User-defined const
import helpers
import const


def extract_zipfile():
    with zipfile.ZipFile(const.ZFILE) as zip_file:
        zip_file.extractall(os.path.join(const.CURRENT_PATH, "dataset"))


def download_data():
    # Download Dataset
    if os.path.isfile(const.ZFILE) or os.path.isfile(os.path.join(const.DATASET_PATH, "All_Ratings.xlsx")):
        print('data already downloaded')
    else:
        print("data does not exist. downloading it.")
        gdown.download(const.DATA_URL, const.ZFILE, quiet=False)
    # Extract ZipFile
    if os.path.isfile(os.path.join(const.DATASET_PATH, "All_Ratings.xlsx")):
        print("data already extracted.")
    else:
        print("extracting data.")
        if not os.path.exists(const.DATA_PATH):
            os.mkdir(os.path.join(const.CURRENT_PATH, "dataset"))
        extract_zipfile()
        # Remove ZipFile
        os.remove(const.ZFILE)


# Download and extract Data
download_data()

# Load NN to detect face
face_cascade = cv2.CascadeClassifier(const.MODEL_PATH)


def getFace(detector, imgPath, imgName):
    imgFullPath = os.path.join(imgPath, imgName)
    img = cv2.imread(imgFullPath)

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
    else:
        # Try resizing still, since our data is kinda normalized
        resized_img = cv2.resize(img, (224, 224))
        print("Error detecting faces, file:" + imgName)

    return resized_img


def randomizeImage(img):
    img = helpers.toimage(img)

    # Rotate Image
    image_rotated = img.rotate(random.random() * 30 - 30)

    # Brightness
    image_brigth = ImageEnhance.Brightness(
        image_rotated).enhance(random.random() * .8 + .6)

    # Contrast
    image_contrast = ImageEnhance.Contrast(
        image_brigth).enhance(random.random() * .6 + .7)

    # Color
    image_color = ImageEnhance.Color(
        image_contrast).enhance(random.random() * .6 + .7)

    randomImg = np.asarray_chkfinite(image_color)

    return randomImg


label_dist = []

# Normalized values in 5 cat.
prVoteImgName = ''
prVoteImgScr1 = 0
prVoteImgScr2 = 0
prVoteImgScr3 = 0
prVoteImgScr4 = 0
prVoteImgScr5 = 0

# Read Labels
ratingFile = open(const.RATING_PATH, 'r')
lines = ratingFile.readlines()
currentIndex = 0

for line in lines:
    line = line.replace('\n', '').split(' ')
    currentIndex += 1
    imgFileName = line[0]
    imgScore = int(float(line[1]))

    # print("Reading Img: " + imgFileName + " Score: " +
    # str(imgScore) + " CIndex: " + str(currentIndex) + "/" + str(lines.__len__()))

    if prVoteImgName == '':
        prVoteImgName = imgFileName

    if (imgFileName != prVoteImgName) or (currentIndex == lines.__len__()):

        totalVotes = prVoteImgScr1 + prVoteImgScr2 + \
            prVoteImgScr3 + prVoteImgScr4 + prVoteImgScr5

        score1 = prVoteImgScr1 / totalVotes
        score2 = prVoteImgScr2 / totalVotes
        score3 = prVoteImgScr3 / totalVotes
        score4 = prVoteImgScr4 / totalVotes
        score5 = prVoteImgScr5 / totalVotes

        im = getFace(face_cascade, const.DATA_PATH, prVoteImgName)

        if isinstance(im, numpy.ndarray):
            normed_img = (im - 127.5) / 127.5

            ld = []
            ld.append(score1)
            ld.append(score2)
            ld.append(score3)
            ld.append(score4)
            ld.append(score5)
            label_dist.append([prVoteImgName, normed_img, ld])

        else:
            print("Error getting face or reading img")

        prVoteImgName = imgFileName
        prVoteImgScr1 = 0
        prVoteImgScr2 = 0
        prVoteImgScr3 = 0
        prVoteImgScr4 = 0
        prVoteImgScr5 = 0

    if imgScore == 1:
        prVoteImgScr1 += 1
    elif imgScore == 2:
        prVoteImgScr2 += 1
    elif imgScore == 3:
        prVoteImgScr3 += 1
    elif imgScore == 4:
        prVoteImgScr4 += 1
    elif imgScore == 5:
        prVoteImgScr5 += 1

ratingFile.close()

# Split data for training + testing
dataSplitIndex = int(label_dist.__len__() - label_dist.__len__()*0.1)

# Shuffle Array
random.shuffle(label_dist)

testLabelDist = label_dist[dataSplitIndex:]
trainLabelDist = label_dist[:dataSplitIndex]

trainDataLen = trainLabelDist.__len__()

# Randomize training data
for i in range(0, trainDataLen):
    img = trainLabelDist[i][1]
    rndImg = randomizeImage(img)
    normedRndImg = (rndImg - 127.5) / 127.5

    trainLabelDist.append([prVoteImgName, normed_img, ld])

# Shuffle and dump data for NN
random.shuffle(trainLabelDist)
pickle.dump(trainLabelDist, open(const.TRAINING_FILE, 'wb'))

random.shuffle(testLabelDist)
pickle.dump(testLabelDist, open(const.TESTING_FILE, 'wb'))
