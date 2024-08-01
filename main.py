import cv2 as cv
import numpy as np

# Capture with camera on computer
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv.imshow("Camera Footage", frame)
    cv.waitKey(1)