import cv2
import face_model
import argparse
import cv2
import sys
import os
import pickle
import numpy as np
from numpy.linalg import norm

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()

class Args():
    def __init__(self):
        self.image_size = '112,112'
        self.gpu = 0
        self.model = '../models/r100-arcface-emore/model,0000'
        self.ga_model = ''
        self.threshold = 1.24
        self.flip = 0
        self.det = 0

args = Args()
model = face_model.FaceModel(args)

with open ('.images/du_lieu_sinh_vien/4/list_feature_mtcnn_lfw.p', 'rb') as fp:
    list_feature = pickle.load(fp)
with open ('.images/du_lieu_sinh_vien/4/name_label_mtcnn_lfw.p', 'rb') as fp:
    list_label = pickle.load(fp)

color=(255, 0, 0)
bcolor=(0,0,255)
thickness=1
font =cv2.FONT_HERSHEY_SIMPLEX
fontScale =1

video_capture = cv2.VideoCapture("http://bao:123456@192.168.100.103:8888/mjpeg") #truyền vào ip camere
img = cv2.imread('')#đường dẫn ảnh


while True:
	ret, frame = video_capture.read()
	try:
		img,bb,lm = model.get_input(frame)
		feature = model.get_feature(img)

		bien = False
		list_dist = []
		list_i = []

		for i in range(len(list_feature)):
			dist = np.sum(np.square(list_feature[i]-feature)) #so sánh feature của ảnh đưa vào với listfeature
			if dist < args.threshold:
					list_dist.append(dist)
					list_i.append(i)
					bien = True
		if bien == True:
			x = min(list_dist)
			y = list_dist.index(x)
			i = list_i[y]
			name = list_label[i]
		else:
			name = "Unknow!"
		acc = (len(list_dist)/10)*100
		box = bb.astype(np.int).flatten()
		cv2.rectangle(frame, (box[0],box[1]), (box[2],box[3]), bcolor,2)
		cv2.putText(frame, str(acc)+"%", (box[2],box[1]), font, fontScale, color, thickness, cv2.LINE_AA)
		cv2.putText(frame, name, (box[0],box[3]), font, fontScale, color, thickness, cv2.LINE_AA)
	except Exception as e:
		frame = frame
	cv2.imshow('Video', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
video_capture.release()
cv2.destroyAllWindows()