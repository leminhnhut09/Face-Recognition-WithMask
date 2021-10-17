import face_model_v2 as face_model
from connection import mysqlconnection
import pymysql.cursors
from datetime import datetime
import argparse
import cv2
import sys
import numpy as np
import pandas as pd
import os
import urllib
import urllib.request
import pickle
import time
parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='../models/r100-arcface-69/model,4', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()
model = face_model.FaceModel(args)

def insertRecord(mssv,log):
    if mssv is not None:
        for x in mssv:
            try :
                connection = mysqlconnection.getConnection()
                cursor = connection.cursor()
                sql =  "Insert into check_logs (MSSV, time) " \
                + " values (%s, %s)"
                # Thực thi sql và truyền 3 tham số
                dt_string = log.strftime("%d/%m/%Y %H:%M:%S")
                print(dt_string)
                cursor.execute(sql,(x,log))
                connection.commit()
            finally:
                connection.close()

with open ('data/list_feature_69_04.p', 'rb') as fp:
    list_feature = pickle.load(fp)
with open ('data/name_label_69_04.p', 'rb') as fp:
    list_label = pickle.load(fp)

color=(255, 0, 0)
bcolor=(0,0,255)
thickness = 1
font =cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1

# VIDEO_STREAM_OUT = "./val/17s_v1.avi"
VIDEO_INPUT = "./image-test/7879.mp4"
vs = cv2.VideoCapture('http://bao:123456@192.168.100.103:8888/mjpeg')
#vs = cv2.VideoCapture(VIDEO_INPUT)
# writer = None
# vs.set(cv2.CAP_PROP_POS_FRAMES, 0)
i = 0
mssv = []
start = time.time()
while True:
    ret, frame = vs.read()
    if not ret:
        print ("Not ret.")
        break
    if i % 5 == 0:
        try:
            imgs,bbox,landmark = model.get_input(frame)
            for img_unit,bbox_unit,landmark_unit in zip(imgs, bbox, landmark):
                f = model.get_feature(img_unit)
                bien = False
                list_dist = []
                list_i = []
                for i in range(len(list_feature)):
                    dist = np.sum(np.square(list_feature[i]-f))
                    if dist < 1.1:
                        list_dist.append(dist)
                        list_i.append(i)
                        bien = True
                if bien==True:
                    x = min(list_dist)
                    if x <= 1.0:
                        y = list_dist.index(x)
                        i = list_i[y]
                        name = list_label[i]

                acc = (len(list_dist)/10)*100
                if acc >= 80:
                    mssv.append(name)
            end = time.time()
            tgian = end-start
            if tgian >= 60:
                start = end
                sv = list(set(mssv))
                now = datetime.now()
                insertRecord(sv,now)
                mssv.clear()
            box=bbox_unit.astype(np.int).flatten()
                # cv2.rectangle(frame, (box[0],box[1]), (box[2],box[3]), bcolor,2)
                # cv2.putText(frame, str(acc)+"%", (box[0],box[1]), font, fontScale, color, thickness, cv2.LINE_AA)
            cv2.putText(frame, name, (box[0],box[3]), font, fontScale, color, thickness, cv2.LINE_AA)
        except Exception as e:
            frame = frame
            # frame = frame
    # if writer is None:
    #     fourcc = cv2.VideoWriter_fourcc(*"XVID")
    #     writer = cv2.VideoWriter(VIDEO_STREAM_OUT, fourcc, 20,(frame.shape[1], frame.shape[0]), True)
    # writer.write(frame)
    i+=1
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("[INFO] cleaning up...")
# writer.release()
vs.release()
cv2.destroyAllWindows()
print("Successfully!")