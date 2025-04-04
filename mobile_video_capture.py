import urllib
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
import matplotlib as plt

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
     min_tracking_confidence = 0.5,
     min_detection_confidence = 0.5 
)

def detect_features(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame.flags.writeable = False
    results = pose.process(frame)
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame, results

def draw_landmarks(frame, results):
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color =(80, 22, 10), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color = (80, 44, 121), thickness=2, circle_radius=2))

url="http://192.168.1.238:8080/video"


cam = cv2.VideoCapture(url) #use 0 for local camera

with mp_pose.Pose(min_detection_confidence= 0.5, min_tracking_confidence=0.5) as pose:
    while(cam.isOpened()):

        ret, frame = cam.read()
        if not ret:
            print("Stream Ended...")
            break
        frame = cv2.resize(frame, (400, 600)) # adjust frame accordingly 
        frame, results = detect_features(frame)
        draw_landmarks(frame, results)
        cv2.imshow('Window 1', frame)
        key = cv2.waitKey(1)
        #if (fps*n)% 18000 == 0: 
            #cv2.imwrite("my_frame {}.jpg".format(i),frame)
            #i+=1
        #n+=1
        if key == ord('q'): # key q to quit the program
            break

 #t_msec = 1000*(minutes*60 + seconds)
        #cam.set(cv2.CAP_PROP_POS_MSEC, t_msec)
        #cv2.imshow('frame', frame); cv2.waitKey(0)
        #cv2.imwrite('my_video_frame.png', frame)