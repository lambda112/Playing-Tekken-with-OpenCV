import cv2 as cv
import numpy as np

# load cascade
face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv.VideoCapture(0)
top_left = (0,0)
bottom_right = (0,0)

def draw_buttons(x, y, image):
    button_1 = cv.rectangle(image, (x + 200, y + 200), (x + 300, y + 300), (0,0,255), 2, cv.LINE_4)
    button_2 = cv.rectangle(image, (x, y + 200), (x - 100, y + 300), (255,0,255), 2, cv.LINE_4)
    button_3 = cv.rectangle(image, (x + 200, y - 100), (x + 300, y), (0,0,255), 2, cv.LINE_4)
    button_4 = cv.rectangle(image, (x, y - 100), (x - 100, y), (255,0,255), 2, cv.LINE_4)

while True:
    ret, image = cap.read()
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(100,100))

    if len(faces):    
        for (x, y, w, h) in faces:
            top_left = (x,y)
            bottom_right = (x + w, y + h)
            image = cv.rectangle(image, top_left, bottom_right, (0,255,0), 2, cv.LINE_4)
            draw_buttons(x,y,image)
    else:
        image = cv.rectangle(image, top_left, bottom_right, (0,255,0), 2, cv.LINE_4)
        draw_buttons(top_left[0], top_left[1], image)


    cv.imshow("Face Detection", image)
    if cv.waitKey(1) == ord("q"):
        cv.destroyAllWindows()
        break

cap.release()


