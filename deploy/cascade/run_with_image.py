import cv2
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
import opencv_model as face_model
import argparse
import numpy as np
import pickle
import time
parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='../../models/r100-arcface-cascade_99/model,0', help='path to load model.')
parser.add_argument('--haarcasecade', default='../haarcascade_frontalface_default.xml', help='path to load haarcasecade.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')

args = parser.parse_args()
model = face_model.FaceModel(args)

with open ('../data/mtcnn_99/list_feature_mtcnn_99.p', 'rb') as fp:
    list_feature = pickle.load(fp)
with open ('../data/mtcnn_99/name_label_mtcnn_99.p', 'rb') as fp:
    list_label = pickle.load(fp)
with open ('../data/mtcnn_99/list_len_mtcnn_99.p', 'rb') as fp:
    list_len_imgs = pickle.load(fp)

color=(255, 0, 0)
bcolor=(0,0,255)
thickness = 1
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1

start = time.time()
image_root = "../image-test/test_image.jpg"
frame = cv2.imread(image_root)
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
                if len(list_dist) >= 10:
                    break
        if bien==True:
            x = min(list_dist)
            y = list_dist.index(x)
            i = list_i[y]
            name = list_label[i]
            num = list_len_imgs[i]
        else:
            name = "Unknow!"
            num = 100
        acc = (len(list_dist)/num)*100
        box=bbox_unit.astype(np.int).flatten()
        cv2.rectangle(frame, (box[0],box[1]), (box[2],box[3]), bcolor,2)
        cv2.putText(frame, str(acc)+"%", (box[0],box[1]), font, fontScale, color, thickness, cv2.LINE_AA)
        cv2.putText(frame, name, (box[0],box[3]), font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow('Image', frame)
    end = time.time()
    interval = end - start
    print(interval)
    cv2.waitKey(0)
except Exception as e:
    print('Error occurred : ' + str(e))

cv2.destroyAllWindows()
print("SuccessFully!")