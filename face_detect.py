import numpy as np
import cv2 
import dlib
import os
import pickle

path = '/Users/thanadonxmac/Documents/Python/Project/'
face_detetor = cv2.CascadeClassifier(f'{path}haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(f'{path}shape_predictor_68_face_landmarks.dat')
model = dlib.face_recognition_model_v1(f'{path}dlib_face_recognition_resnet_model_v1.dat')
FACE_DESC, FACE_NAME = pickle.load(open('trainset.pk','rb'))

cap = cv2.VideoCapture(0)

while(True):
    _,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_detetor.detectMultiScale(gray,1.3,5)

    for (x, y, w, h) in faces:
        img = frame[y-10:y+h+10, x-10:x+w+10][:,:,::-1]
        dets = detector(img, 1)

        for k, d in enumerate(dets):
                shape = sp(img, d)
                face_desc0 = model.compute_face_descriptor(img, shape, 100)
                d = []
                for face_desc in FACE_DESC:
                    d.append(np.linalg.norm(np.array(face_desc) - np.array(face_desc0)))
                d = np.array(d)
                idx = np.argmin(d)
                if d[idx] < .5:
                    name = FACE_NAME[idx]
                    print(name)


        

        


