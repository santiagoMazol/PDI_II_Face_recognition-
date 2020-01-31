# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 16:55:16 2019

@author: fkevin.castro
"""

import os 
import numpy as np
#from PIL import Image
import cv2
import pickle


                

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "train_images")
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root,file)
            label = os.path.basename(root).replace(" ","-").lower()
            #y_labels.append(label)
            #x_train.append(path)
            
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
                
            id_ =label_ids[label]
            pil_image = cv2.imread(path)            
            pil_image = cv2.cvtColor(pil_image,cv2.COLOR_BGR2GRAY)
            size= (200,200)
            final_image = cv2.resize(pil_image,size)
            image_array = np.array(final_image, "uint8")
            
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

            
            for (x,y,w,h) in faces:
                roi = image_array[y:y+h,x:x+h]
                cv2.namedWindow("Rostros", 1)
                cv2.imshow(label,roi)
                cv2.waitKey(0)
                x_train.append(roi)
                y_labels.append(id_)
    
with open("labels.pickle",'wb') as f:
    pickle.dump(label_ids, f)
    
recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner.yml")


