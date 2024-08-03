import cv2 as cv
import numpy as np
import time

# load cascade
face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv.VideoCapture(0)
ret, current_frame = cap.read()
previous_frame = current_frame

top_left = (0,0)
bottom_right = (0,0)

# Initialize cooldown tracking for each button
button_cooldowns = {0: time.time(), 1: time.time(), 2: time.time(), 3: time.time()}
COOLDOWN = 2  # Cooldown period in seconds

def perform_action(id):
    if id == 0:
        print("button 0 pressed")

    if id == 1:
        print("button 1 pressed")

    if id == 2:
        print("button 2 pressed")

    if id == 3:
        print("button 3 pressed")

    current_time = time.time()


def draw_buttons(x, y, current_gray, previous_gray):
    global start_time
    buttons = [
        ((x + 200, y + 100), (x + 300, y + 200)), # button 0
        ((x - 200, y + 100), (x - 100, y+200)), # button 1
        ((x + 200, y - 100), (x + 300, y)), # button 2
        ((x - 200, y - 100), (x - 100, y)) # button 3
    ]

    frame_height, frame_width, _ = current_gray.shape

    for index, (topl, botr) in enumerate(buttons):
        topl_x = max(0, topl[0])
        topl_y = max(0, topl[1])
        botr_x = min(frame_width, botr[0])
        botr_y = min(frame_height, botr[1])

        if topl_x < botr_x and topl_y < botr_y: 
            cv.rectangle(current_gray, (topl_x, topl_y), (botr_x, botr_y), (255, 0, 255), 2, cv.LINE_4)
            cv.putText(current_gray, f"button{index}", (topl_x,topl_y-5), cv.FONT_HERSHEY_COMPLEX,0.6,(0,255,0), 1)
            
            SKIN_COLOR_LOW = np.array([3, 30, 50], dtype=np.uint8)  # Lower bound for brown skin
            SKIN_COLOR_HIGH = np.array([30, 150, 200], dtype=np.uint8)  # Upper bound for brown skin
            
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

            diff_mask = cv.bitwise_xor(mask_current, mask_previous)
            diff_abs = cv.absdiff(mask_current, mask_previous)
            combined_diff = cv.add(diff_abs, diff_mask)
            current_time = time.time()

            if np.sum(combined_diff) > 500000 and current_time - button_cooldowns[index] > COOLDOWN:    
                cv.rectangle(current_gray, (topl_x, topl_y), (botr_x, botr_y), (0, 255, 0), 2, cv.LINE_4)
                perform_action(index)
                button_cooldowns[index] = current_time

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


