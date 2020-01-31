# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 15:20:57 2019

@author: fkevin.castro
"""

import numpy as np
import pickle
import cv2

cv2.destroyAllWindows()
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels ={"person_name":1}
with open("labels.pickle",'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for(x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+h]
        id_, conf = recognizer.predict(roi_gray)


        font = cv2.FONT_HERSHEY_SIMPLEX
        name = labels[id_]
        color =(255,0,0)
        stroke= 2
        cv2.putText(frame, name, (x,y), font ,1,color,stroke,cv2.LINE_AA)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    #Guardar cara en un archivo aparte
       # roi_color =frame[y:y+h, x:x+h]
       # cv2.imwrite("cara.png",roi_color)

    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllwindows()