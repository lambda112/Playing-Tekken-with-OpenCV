import cv2 as cv
import numpy as np
import time
import direct_input as pydirectinput
import threading
from button_config import button_block

PADDING = 0

# load cascade
face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv.VideoCapture(0)
time.sleep(1)

ret, current_frame = cap.read()
previous_frame = current_frame.copy()
top_left = (0,0)
bottom_right = (0,0)
x,y,w,h = 0,0,0,0

# Initialize cooldown tracking for each buttons
button_cooldowns = {0: time.time(), 1: time.time(), 2: time.time(), 3: time.time(), 4:time.time(), 5:time.time()}

COOLDOWN = 2  # Cooldown period in seconds

def find_nonzero_num(topl_y, botr_y, topl_x, botr_x, current_frame, previous_frame):
    roi_current = current_frame[topl_y:botr_y, topl_x:botr_x]
    roi_previous = previous_frame[topl_y:botr_y, topl_x:botr_x]

    # Convert the ROI to HSV color space
    roi_current_gray = cv.GaussianBlur(roi_current, (5,5), 0)
    roi_previous_gray = cv.GaussianBlur(roi_previous, (5,5),0)

    # Convert the ROI to HSV color space
    roi_current_gray = cv.cvtColor(roi_current_gray, cv.COLOR_BGR2GRAY)
    roi_previous_gray = cv.cvtColor(roi_previous_gray, cv.COLOR_BGR2GRAY)

    diff_abs = cv.absdiff(roi_current_gray, roi_previous_gray)
    _, threshold = cv.threshold(diff_abs, 20, 255, cv.THRESH_BINARY)

    threshold = cv.dilate(threshold, (5,5), iterations=1)
    threshold = cv.erode(threshold, (5,5), iterations=1)
    non_zero = np.count_nonzero(threshold)
    return non_zero

def perform_action(states: dict, action):
    pydirectinput.PAUSE = 0.01
    button_true = {k:v for k,v in states.items() if v == True}
    action = {k:action[k] for k,v in button_true.items()}    
    action = list(action.values())

    pydirectinput.hotkey(interval=0.5, *action)
    print(button_true)
    
def draw_buttons(x, y, current_frame, previous_frame, face_topl, face_botr):
    buttons = button_block(x,y, "old")
    button_state = {0: False, 1: False, 2: False, 3: False, 4:False, 5:False}
    BUTTON_ACTION = ["j", "i", "k", "l", "a", "d"]
    frame_height, frame_width, _ = current_frame.shape

    for index, (topl, botr) in buttons.items():
        topl_x = max(0, topl[0])
        topl_y = max(0, topl[1])
        botr_x = min(frame_width, botr[0])
        botr_y = min(frame_height, botr[1])

        if topl_x < botr_x and topl_y < botr_y: 
            cv.rectangle(current_frame, (topl_x, topl_y), (botr_x, botr_y), (255, 0, 255) if index > 4 else (255,0,0), 2, cv.LINE_4)
            cv.putText(current_frame, f"button{index}", (topl_x+20,topl_y+90) if index < 4 else (topl_x,topl_y-5), cv.FONT_HERSHEY_COMPLEX,0.6,(0,255,0), 1)
            non_zero = find_nonzero_num(topl_y, botr_y, topl_x, botr_x, current_frame, previous_frame)

            if non_zero > 1000:
                button_state[index] = True
                cv.rectangle(current_frame, (topl_x, topl_y), (botr_x, botr_y), (0, 255, 0) if index < 4 else (255, 0, 255), 2, cv.LINE_4)
        
                if index == 4:
                    # left
                    if face_topl[0] < botr_x:
                        button_state[index] = True
                        cv.rectangle(current_frame, (topl_x, topl_y), (botr_x, botr_y), (0, 255, 255), lineType=cv.LINE_4)
                    else:
                        button_state[index] = False
                    
                if index == 5:
                    # right
                    if face_botr[0] > topl_x:
                        button_state[index] = True
                        cv.rectangle(current_frame, (topl_x, topl_y), (botr_x, botr_y), (0, 255, 255), lineType=cv.LINE_4)
                    else:
                        button_state[index] = False
                
    print(" ")
    threading.Thread(target = perform_action, args=(button_state, BUTTON_ACTION)).start()
    button_state = {0: False, 1: False, 2: False, 3: False, 4:False, 5:False}          

while True:
    current_frame = cv.flip(current_frame, 1)
    faces = face_cascade.detectMultiScale(current_frame, 1.1, 3, minSize=(40,40), maxSize=(400,400))

    if len(faces):    
        for (x, y, w, h) in faces:
            x,y,w,h = x,y,w,h
            top_left = (x + PADDING ,y+PADDING)
            bottom_right = ((x + w) - PADDING, (y + h) - PADDING)
            cv.rectangle(current_frame, top_left, bottom_right, (0,255,0), 4, cv.LINE_4)

    draw_buttons(current_frame.shape[1], current_frame.shape[0], current_frame, previous_frame, top_left, bottom_right)
    cv.imshow("Face Detection", current_frame)
    if cv.waitKey(1) == ord("q") & 0xFF:
        break
    
    top_left = (x,y)
    bottom_right = (x + w, y + h)
    previous_frame = current_frame.copy()
    ret, current_frame = cap.read()

cv.destroyAllWindows()
cap.release()


