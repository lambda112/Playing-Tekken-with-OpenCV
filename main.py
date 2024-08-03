import cv2 as cv
import numpy as np

# load cascade
face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv.VideoCapture(0)
ret, current_frame = cap.read()
previous_frame = current_frame

top_left = (0,0)
bottom_right = (0,0)

def perform_action(id):
    if id == 0:
        print("button 1 pressed")

    if id == 1:
        print("button 2 pressed")

    if id == 0:
        print("button 3 pressed")

    if id == 1:
        print("button 4 pressed")


def draw_buttons(x, y, current_gray, previous_gray):
    buttons = [
        ((x + 100, y + 100), (x + 200, y + 200)), # button 0
        ((x - 100, y + 100), (x, y+200)), # button 1
        ((x + 100, y - 100), (x + 200, y)), # button 2
        ((x - 100, y - 100), (x, y)) # button 3
    ]

    frame_height, frame_width, _ = current_gray.shape

    for index, (topl, botr) in enumerate(buttons):
        topl_x = max(0, topl[0])
        topl_y = max(0, topl[1])
        botr_x = min(frame_width, botr[0])
        botr_y = min(frame_height, botr[1])

        if topl_x < botr_x and topl_y < botr_y: 
            cv.rectangle(current_gray, (topl_x, topl_y), (botr_x, botr_y), (255, 0, 255), 2, cv.LINE_4)
            cv.putText(current_gray, f"button{index}", (topl_x,topl_y-5), cv.FONT_HERSHEY_COMPLEX,0.6,(255,255,0), 1)
            
            SKIN_COLOR_LOW = np.array([0, 20, 50], dtype=np.uint8)  # Lower bound for brown skin
            SKIN_COLOR_HIGH = np.array([20, 200, 200], dtype=np.uint8)  # Upper bound for brown skin
            
            roi_current = current_gray[topl_y:botr_y, topl_x:botr_x]
            roi_previous = previous_gray[topl_y:botr_y, topl_x:botr_x]

            roi_current_blur = cv.GaussianBlur(roi_current, (5, 5), 0)
            roi_previous_blur = cv.GaussianBlur(roi_previous, (5, 5), 0)

            # Convert the ROI to HSV color space
            roi_current_hsv = cv.cvtColor(roi_current_blur, cv.COLOR_BGR2HSV)
            roi_previous_hsv = cv.cvtColor(roi_previous_blur, cv.COLOR_BGR2HSV)

            # Create masks for skin color
            mask_current = cv.inRange(roi_current_hsv, SKIN_COLOR_LOW, SKIN_COLOR_HIGH)
            mask_previous = cv.inRange(roi_previous_hsv, SKIN_COLOR_LOW, SKIN_COLOR_HIGH)

            diff = cv.absdiff(mask_current, mask_previous)
            if np.sum(diff) > 400000:
                cv.rectangle(current_gray, (topl_x, topl_y), (botr_x, botr_y), (0, 255, 0), 2, cv.LINE_4)
                perform_action(index)

while True:
    faces = face_cascade.detectMultiScale(current_frame, 1.1, 5, minSize=(40,40))

    if len(faces):    
        for (x, y, w, h) in faces:
            top_left = (x,y)
            bottom_right = (x + w, y + h)
            cv.rectangle(current_frame, top_left, bottom_right, (0,255,0), 2, cv.LINE_4)
            draw_buttons(x,y,current_frame, previous_frame)
    else:
        cv.rectangle(current_frame, top_left, bottom_right, (0,255,0), 2, cv.LINE_4)
        draw_buttons(top_left[0], top_left[1], current_frame, previous_frame)


    cv.imshow("Face Detection", current_frame)
    if cv.waitKey(1) == ord("q"):
        cv.destroyAllWindows()
        break

    previous_frame = current_frame.copy()
    ret, current_frame = cap.read()

cap.release()


