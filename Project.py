import numpy as np
import cv2 
import dlib
import pickle
import pandas
from time import  localtime, strftime
from oauth2client.service_account import ServiceAccountCredentials
import gspread
from df2gspread import df2gspread as d2g

path = '/Users/thanadonxmac/Documents/Python/Project/'
face_detetor = cv2.CascadeClassifier(f'{path}haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(f'{path}shape_predictor_68_face_landmarks.dat')
model = dlib.face_recognition_model_v1(f'{path}dlib_face_recognition_resnet_model_v1.dat')
FACE_DESC, FACE_NAME = pickle.load(open('trainset.pk','rb'))

scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']

credentials = ServiceAccountCredentials.from_json_keyfile_name('liststudent.json', scope)
gc = gspread.authorize(credentials)
spreadsheet_key = '1BiGryQk5HHiVo1lnTg4r7xS4mIEwWZsT5uYGy-uEz0E'

day = ""
data = []
num = 0

checked = []

def check_user(check):
    global day,data,num
    
    if strftime("%a", localtime()) != day:
        day = strftime("%a", localtime())
        data = pandas.read_csv(f'{path}data/learn_{day}.csv').T
    
    for i in range(len(data.index)):
        if i > 1:
            learn = data[0][i]
            if learn == "พักเที่ยง":
                continue
            if  int(strftime("%H", localtime())) > int(learn[0:2])-1 and int(strftime("%H", localtime())) < int(learn[6:8]):
                num = i

    if num == 0:
        return 

    find = 0
    data_np = np.array(data)
    for count in data_np[1]:
        if count == check:
            if count not in checked:
                data[find][num] = (strftime("%H:%M:%S", localtime()))
                goto_csv(data)
                checked.append(count)
        else:
            find += 1

def goto_csv(read_data):
    name_file = strftime("%a %d %b %Y", localtime())
    read_data.T.to_csv(f"/Users/thanadonxmac/Documents/Python/Project/data_check/{name_file}.csv")

    d2g.upload(read_data.T, spreadsheet_key, name_file, credentials=credentials, row_names=True)

def detect():
    _,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_detetor.detectMultiScale(gray,1.3,5)

    for (x, y, w, h) in faces:
        img = frame[y-10:y+h+10, x-10:x+w+10][:,:,::-1]
        dets = detector(img, 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

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
                check_user(name)
                print(name)

    cv2.imshow("Frame", frame)
    cv2.waitKey(1)

while (True):
    detect()