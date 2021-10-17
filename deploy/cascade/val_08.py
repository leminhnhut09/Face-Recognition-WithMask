import cv2
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
import opencv_model as face_model
import argparse
import numpy as np
import pickle
from imutils.video import VideoStream
parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='../../recognition/models/374_09/model,1', help='path to load model.')
parser.add_argument('--haarcasecade', default='../haarcascade_frontalface_default.xml', help='path to load haarcasecade.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()
model = face_model.FaceModel(args)

with open ('../data/374_09/list_feature_mtcnn_lfw.p', 'rb') as fp: 
    list_feature = pickle.load(fp)
with open ('../data/374_09/name_label_mtcnn_lfw.p', 'rb') as fp:   
    list_label = pickle.load(fp)

color=(255, 255, 0)
bcolor=(0,0,255)
thickness = 1
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1

count = 0
count_detect = 0
num = 0
# đường dẫn lấy ảnh
imgs_dir_val = r'D:\TruongCongHau\KhoaLuanTotNghiep_2021\khoaluantotnghiep\deploy\images\du_lieu_sinh_vien\374_09\test_nomask'
folders = os.listdir(imgs_dir_val)  # data
for folder in folders:  # anh
    imgs_list = os.listdir(os.path.join(imgs_dir_val, folder))  # lst img
    num += len(imgs_list)
    path = imgs_dir_val + '\\' + folder
    for pos in range(len(imgs_list)):
        link = path +  '\\' +  imgs_list[pos]
        frame = cv2.imread(link)
        try:
           imgs,bbox,landmark = model.get_input(frame)
           for img_unit,bbox_unit,landmark_unit in zip(imgs, bbox, landmark):
             count_detect += 1
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
             else:
               name = "Unknow!"
               acc = (len(list_dist)/10)*100
             #check
             if(folder.strip() == name.strip()):
                     count += 1
             print(name, '-', folder, '-', count)    
            
        except:
          pass

print("Số ảnh detect được", count_detect)
print("Số ảnh đúng", count)
print("Tổng ảnh", num)
print("Độ chinh xác", count / num)

