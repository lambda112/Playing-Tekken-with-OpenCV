import cv2 as cv
import numpy as np
import time
import direct_input as pydirectinput
import mediapipe as mp 
from multiprocessing.pool import ThreadPool
from button_config import button_block

mp_face = mp.solutions.face_mesh
mp_face_mesh = mp_face.FaceMesh(max_num_faces = 1, refine_landmarks = True, min_detection_confidence = 0.3, min_tracking_confidence = 0.3)
mp_drawing = mp.solutions.drawing_utils

# set up
cap = cv.VideoCapture(0)
ret, frame = cap.read()
previous_frame = frame.copy()
time.sleep(1)

# pools 
action_pool = ThreadPool(processes=2)
camera_pool = ThreadPool(processes=1)

PADDING = 0
BUTTON_ACTION = ["j", "i", "k", "l", "a", "d", "s", "w"]

def perform_action(states: dict, action):
    pydirectinput.PAUSE = 0.003
    print(states)
    button_true = {k:v for k,v in states.items() if v == True}
    print(button_true)

    action = {k:action[k] for k,v in button_true.items()}    
    action = list(action.values())

    if action:
        for key in action: 
            pydirectinput.keyDown(key)
    
    else:
        for key in BUTTON_ACTION:
            pydirectinput.keyUp(key)
    
    
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

def draw_face(index, w, h, topl, botr, button_state):
    results = mp_face_mesh.process(frame)  
    if results.multi_face_landmarks:
        
        for i, pos in enumerate(results.multi_face_landmarks[0].landmark):
            if i in [5]:
                cx, cy = int(pos.x * w), int(pos.y * h)
                cv.circle(frame, (cx, cy), 50, (255, 0, 0), 1) 

                if index == 4:
                    # left
                    if botr[0] > cx - 20:
                        button_state[index] = True
                        cv.rectangle(frame, (topl[0], topl[1]), (botr[0], botr[1]), (0, 255, 255), lineType=cv.LINE_4)
                    else:
                        button_state[index] = False
                        
                    
                elif index == 5:
                    # right
                    if topl[0] < cx + 20:
                        button_state[index] = True
                        cv.rectangle(frame, (topl[0], topl[1]), (botr[0], botr[1]), (0, 255, 255), lineType=cv.LINE_4)
                    else:
                        button_state[index] = False
                
                elif index == 6:
                    # down
                    if topl[1] < cy:
                        button_state[index] = True
                        cv.line(frame, topl, botr, (0,0,255), 4, cv.LINE_4)
                    else:
                        button_state[index] = False
                
                elif index == 7:
                    # down
                    if topl[1] > cy:
                        button_state[index] = True
                        cv.line(frame, topl, botr, (0,0,255), 4, cv.LINE_4)
                    else:
                        button_state[index] = False
                
                else:
                    button_state[index] = False
  
    
def draw_items(w, h, frame, previous_frame):
    buttons = button_block(w,h, "old")
    button_state = {0: False, 1: False, 2: False, 3: False, 4:False, 5:False}

    for index, (topl, botr) in buttons.items():

        if index < 6:
            cv.rectangle(frame, (topl[0], topl[1]), (botr[0], botr[1]), (255, 0, 255) if index > 4 else (255,0,0), 2, cv.LINE_4)
            cv.putText(frame, f"button{index}", (topl[0]+20,topl[1]+90) if index < 4 else (topl[0],topl[1]-5), cv.FONT_HERSHEY_COMPLEX,0.6,(0,255,0), 1)

            if index < 4:
                non_zero = find_nonzero_num(topl[1], botr[1], topl[0], botr[0], frame, previous_frame)
                if non_zero > 1000:
                    button_state[index] = True
            
            else:
                draw_face(index, w, h, topl, botr, button_state)

        else:
            cv.line(frame, topl, botr, (0,255,0), 4, cv.LINE_4)
            draw_face(index, w, h, topl, botr, button_state)
                        

    action_pool.apply_async(perform_action, args=(button_state, BUTTON_ACTION))

def get_camera():
    ret, frame = cap.read()
    return ret, frame

while True:
    frame = cv.flip(frame, 1)
    frame = cv.GaussianBlur(frame, (1,1), 0)
    frame = cv.dilate(frame, (1,1), iterations=1)
    frame = cv.erode(frame, (1,1), iterations=1)
    draw_items(frame.shape[1], frame.shape[0], frame, previous_frame)

    cv.imshow("Face Detection", frame)
    if cv.waitKey(1) == ord("q") & 0xFF:
        break
    
    previous_frame = frame.copy()
    camera_result = camera_pool.apply_async(get_camera)
    ret, frame = camera_result.get()

    
cv.destroyAllWindows()
cap.release()
