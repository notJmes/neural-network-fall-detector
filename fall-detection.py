import cv2
import mediapipe as mp
import time
import os
import tensorflow as tf
from numpy import average
import numpy as np

cwd = os.path.dirname(os.path.realpath(__file__))

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

try:
    model = tf.keras.models.load_model(os.path.join(cwd,'saved_model/fall_detection_model.h5'))
except Exception as e:
    print('failed to import model:', e)
    exit()
pTime = 0
predict = 0
fall = False
while True:
    predict += 1
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    coords = {'12': (0,0,0) , # shoulder R
              '11': (0,0,0) , # shoulder L
              '24': (0,0,0) , # feet R
              '23': (0,0,0) } # feet L
    temp = []
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS, )
        for id, lm in enumerate(results.pose_landmarks.landmark):
            [temp.append(val) for val in [lm.x, lm.y, lm.z]]

    

    if predict >= 30:
        try:
            prediction = model.predict([temp])
            if np.argmax(prediction)==1:
                fall = True
            else:
                fall = False
        except:
            fall = False
        predict = 0

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)

    if not fall:
        cv2.putText(img, str('No fall detected'), (150, 50), cv2.FONT_HERSHEY_PLAIN, 2,
                    (0, 255, 0), 2)
    else:
        cv2.putText(img, str('Fall detected'), (150, 50), cv2.FONT_HERSHEY_PLAIN, 2,
                    (0, 0, 255), 2)

    name = ['Shoulder R', 'Shoulder L', 'Feet R', 'Feet L']
    for i, id in enumerate(coords):
        cv2.putText(img, name[i]+str(coords[id]), (70, 60+60*(i+1)), cv2.FONT_HERSHEY_PLAIN, 1,
                    (255, 255, 0), 1)

    cv2.imshow('Viewer', img)

    cv2.waitKey(15)

# https://www.youtube.com/watch?v=brwgBf6VB0I