# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 15:20:57 2019

@author: fkevin.castro
"""

import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for(x,y,w,h) in faces:
         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #Guardar cara en un archivo aparte
       # roi_color =frame[y:y+h, x:x+h]
       # cv2.imwrite("cara.png",roi_color)
        
    cv2.imshow('frame',frame)
    if cv2.waitKey(27) & 0xFF == ord('q'):
        break
    

cap.release()
cv2.destroyAllwindows()