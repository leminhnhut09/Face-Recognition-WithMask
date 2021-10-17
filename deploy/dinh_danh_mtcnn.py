import face_model
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
        self.model = '../recognition/models/374_08/model,0001'#đường dẫn đến model vừa tạo
        self.ga_model = ''  #lay tuổi + gt
        self.threshold = 1.24
        self.flip = 0
        self.det = 0

args = Args()
model = face_model.FaceModel(args)
imgs_dir = './images/du_lieu_sinh_vien/374_08/train' # đường dẫn hình
folders = os.listdir(imgs_dir)
list_feature = []
name_label = []
list_len = []
start = time.time()
for folder in folders:
    imgs = os.listdir(os.path.join(imgs_dir,folder))
    for img in imgs:
        bien = False
        img_root_path = os.path.join(imgs_dir,folder,img)
        print(img_root_path)
        img = cv2.imread(img_root_path)
        if img is not None:
            try:
                img,bb,lm = model.get_input(img)
                img = model.get_feature(img)
                list_feature.append(img)
                name_label.append(folder)
                bien = True
            except Exception as e:
                print('Error occurred : ' + str(e))
        else:
            print("Ko doc duoc hinh anh: ",img_root_path)
        if bien == True:
            list_len.append(len(imgs))
            
with open('./images/du_lieu_sinh_vien/374_08/list_len_mtcnn_lfw.p', 'wb') as fp: #list lưu số lượng hình của 1 thư mục
    pickle.dump(list_len, fp)
    
with open('./images/du_lieu_sinh_vien/374_08/list_feature_mtcnn_lfw.p', 'wb') as fp: #vị trí lưu listfeature
    pickle.dump(list_feature, fp)
    
with open('./images/du_lieu_sinh_vien/374_08/name_label_mtcnn_lfw.p', 'wb') as fp: #list lưu tên thư mục
    pickle.dump(name_label, fp)
    
end = time.time()

interval = end - start
print("Total time: %s "%str(interval))
"""with open ('data/list_feature.p', 'rb') as fp:
    list_feature = pickle.load(fp)
with open ('data/name_label.p', 'rb') as fp:
    list_label = pickle.load(fp)"""
#hinh anh dua vao
"""img3 = cv2.imread('./Giang_0003.JPG')
img3,bb3,lm3 = model.get_input(img3)
f3 = model.get_feature(img3)
print("f3: ",f3[0:10])
print("threshold: ",args.threshold)
bien = False
list_dist = []
list_i = []
for i in range(len(list_feature)):
    dist = np.sum(np.square(list_feature[i]-f3))
    if dist < args.threshold:
        list_dist.append(dist)
        list_i.append(i)
        bien = True
    
if bien==True:
    x = min(list_dist)
    y = list_dist.index(x)
    i = list_i[y]
    print("Nguoi trong hinh la: ", list_label[i])
else:
    print("Unknow!!!")"""

print("Success!")