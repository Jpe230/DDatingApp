import cv2
import nn.neuralnet as nn

TEST_PHOTO = []

for i in TEST_PHOTO:

    vc=cv2.VideoCapture(i)
    _, image=vc.read()

    nn.beauty_predict(_,image)