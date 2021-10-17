import opencv_model
import argparse
import cv2
import sys
import os
import pickle
import time
import numpy as np
from numpy.linalg import norm

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='', help='path to load model.')
parser.add_argument('--haarcasecade', default='', help='path to load haarcasecade.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()

class Args():
    def __init__(self):
        self.image_size = '112,112'
        self.gpu = 0
        self.model = '../models/r100-arcface-cascade_99/model,0000'
        self.haarcasecade = './haarcascade_frontalface_default.xml'
        self.threshold = 1.1
        self.flip = 0
        self.det = 0

args = Args()
model = opencv_model.FaceModel(args)
imgs_dir = './images/images/'
folders = os.listdir(imgs_dir)
list_feature = []
name_label = []
list_len_image = []
start = time.time()
for folder in folders:
    imgs = os.listdir(os.path.join(imgs_dir,folder))
    for img in imgs:
        bien = False
        img_root_path = os.path.join(imgs_dir,folder,img)
        print(img_root_path)
        img = cv2.imread(img_root_path)
        try:
            img,bb,lm = model.get_input_v2(img)
            img = model.get_feature(img)
            list_feature.append(img)
            name_label.append(folder)
            bien = True
        except Exception as e:
            print('Error occurred : ' + str(e))
        if bien == True:
            list_len_image.append(len(imgs))


with open('data/list_len_cascade_99.p','wb') as li:
    pickle.dump(list_len_image,li)
with open('data/list_feature_cascade_99.p', 'wb') as lf:
    pickle.dump(list_feature, lf)
with open('data/name_label_cascade_99.p', 'wb') as lb:
    pickle.dump(name_label, lb)
print("Success!!!")
# with open ('data/list_feature_v1.p', 'rb') as fp:
#     list_feature = pickle.load(fp)
# with open ('data/name_label_v1.p', 'rb') as fp:
#     list_label = pickle.load(fp)

# #hinh anh dua vao
# img3 = cv2.imread('./Giang_0003.JPG')
# img3,bb3,lm3 = model.get_input(img3)
# f3 = model.get_feature(img3)
# print(bb3)
# print(lm3)
# print("f3: ",f3[0:10])
# print("threshold: ",args.threshold)
# bien = False
# list_dist = []
# list_i = []
# for i in range(len(list_feature)):
#     dist = np.sum(np.square(list_feature[i]-f3))
#     print("dist: ", dist)
#     if dist < args.threshold:
#         list_dist.append(dist)
#         list_i.append(i)
#         bien = True
    
# if bien==True:
#     x = min(list_dist)
#     y = list_dist.index(x)
#     i = list_i[y]
#     print("Nguoi trong hinh la: ", list_label[i])
# else:
#     print("Unknow!!!")

    