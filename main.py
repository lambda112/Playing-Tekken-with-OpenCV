import cv2 as cv
import numpy as np
import time
import pydirectinput

pydirectinput.PAUSE=0.05

# load cascade
#time.sleep(3)
face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv.VideoCapture(0)
ret, current_frame = cap.read()
previous_frame = current_frame.copy()

top_left = (0,0)
bottom_right = (0,0)

# Initialize cooldown tracking for each buttonxzazxx
button_cooldowns = {0: time.time(), 1: time.time(), 2: time.time(), 3: time.time()}
COOLDOWN = 2  # Cooldown period in secondsxzx

def perform_action(id, face_topl, face_botr, button_topl, button_botr):
    if id == 4:
        if face_botr[0] < button_botr[0]:
            cv.rectangle(current_frame, (button_topl[0], button_topl[1]), (button_botr[0], button_botr[1]), (0, 255, 0))
            print(face_botr, button_botr)
            pydirectinput.press("left") 

    # action = [lambda: pydirectinput.press("x", interval=0.1), lambda: pydirectinput.press("z", interval=0.1), lambda: pydirectinput.press("s", interval=0.1), lambda: pydirectinput.press("a", interval=0.1)]
    # action[id]()
    # print(f"button: {id}")
    pass

def draw_buttons(x, y, current_frame, previous_frame, face_topl, face_botr):
    # 480 640
    buttons = [
        ((x - 150, 0 + 50), (x - 50, 0 + 150)), # button 0
        ((x - 600, y - 150), (x - 500, y - 50)), # button 1
        ((0 + 40, 0 + 50), (0 + 140, 0 + 150)),       # button 2
        ((x - 150, y - 150), (x - 50, y - 50)), # button 3
        ((0 + 140, 0 + 50), (0 + 240, 0 + 150)), # button 4
        # ((x - 150, y - 150), (x - 50, y - 50)), # button 5
    ]

    frame_height, frame_width, _ = current_frame.shape

    for index, (topl, botr) in enumerate(buttons):
        topl_x = max(0, topl[0])
        topl_y = max(0, topl[1])
        botr_x = min(frame_width, botr[0])
        botr_y = min(frame_height, botr[1])

        if topl_x < botr_x and topl_y < botr_y: 
            cv.rectangle(current_frame, (topl_x, topl_y), (botr_x, botr_y), (255, 0, 255), 2, cv.LINE_4)
            cv.putText(current_frame, f"button{index}", (topl_x,topl_y-5), cv.FONT_HERSHEY_COMPLEX,0.6,(0,255,0), 1)

            roi_current = current_frame[topl_y:botr_y, topl_x:botr_x]
            roi_previous = previous_frame[topl_y:botr_y, topl_x:botr_x]

            # Convert the ROI to HSV color space
            roi_current_gray = cv.cvtColor(roi_current, cv.COLOR_BGR2GRAY)
            roi_previous_gray = cv.cvtColor(roi_previous, cv.COLOR_BGR2GRAY)

            diff_abs = cv.absdiff(roi_current_gray, roi_previous_gray)
            _, threshold = cv.threshold(diff_abs, 25, 255, cv.THRESH_BINARY)
            non_zero = np.count_nonzero(threshold)

            # current_time = time.time()
            if np.sum(non_zero) > 1000: # and current_time - button_cooldowns[index] > COOLDOWN:  
                cv.rectangle(current_frame, (topl_x, topl_y), (botr_x, botr_y), (0, 255, 0) if index < 4 else (255, 0, 255), 2, cv.LINE_4)
                perform_action(index, face_topl, face_botr, (topl_x, topl_y), (botr_x, botr_y))
                # button_cooldowns[index] = current_time
                print(f"Button {index} - Difference sum: {np.sum(diff_abs)}")  

while True:
    faces = face_cascade.detectMultiScale(current_frame, 1.1, 5, minSize=(40,40))
    if len(faces):    
        for (x, y, w, h) in faces:
            top_left = (x,y)
            bottom_right = (x + w, y + h)
            cv.rectangle(current_frame, top_left, bottom_right, (0,255,0), 2, cv.LINE_4)
    else:
        cv.rectangle(current_frame, top_left, bottom_right, (0,255,0), 2, cv.LINE_4)
        
    draw_buttons(current_frame.shape[1], current_frame.shape[0], current_frame, previous_frame, top_left, bottom_right)


    cv.imshow("Face Detection", current_frame)
    if cv.waitKey(1) == ord("q"):
        break

    previous_frame = current_frame.copy()
    ret, current_frame = cap.read()

cv.destroyAllWindows()
cap.release()


