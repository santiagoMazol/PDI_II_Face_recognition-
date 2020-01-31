# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 15:20:57 2019

@author: fkevin.castro
"""
import os
import numpy as np
import pickle
import cv2

cv2.destroyAllWindows()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "faces")

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels ={"person_name":1}
with open("labels.pickle",'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root,file)
            frame = cv2.imread(path)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
           
            for(x,y,w,h) in faces:
                roi_gray = gray[y:y+h, x:x+h]
                id_, conf = recognizer.predict(roi_gray)
                print(id_,conf)
                if conf <80 and conf>45:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    name = labels[id_]
                    color =(255,0,0)
                    stroke= 2
                    cv2.putText(frame, name, (x,y), font ,1,color,stroke,cv2.LINE_AA)
            
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            cv2.imshow(name,frame)
            cv2.waitKey(0)
      
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()