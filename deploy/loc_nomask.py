import cv2
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
import opencv_model as face_model
import argparse
import numpy as np
import pickle
from imutils.video import VideoStream
import shutil
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


# đường dẫn lấy ảnh
imgs_dir_val = r'C:\Users\CongHau\Desktop\Data_GuiThayHa\DATA_374\AFDB_face_dataset'
img_dir_out = r'C:\Users\CongHau\Desktop\Data_GuiThayHa\DATA_374\nomask_them'
img_dir_out_num = r'C:\Users\CongHau\Desktop\Data_GuiThayHa\DATA_374\mask'
folders = os.listdir(imgs_dir_val)  # data


for folder in folders:  # anh
    imgs_list = os.listdir(os.path.join(imgs_dir_val, folder))  # lst img
    
    os.mkdir(img_dir_out + '\\' + folder)
    sum = 0
    path = imgs_dir_val + '\\' + folder
    path_out = img_dir_out + '\\' + folder
    num = len(os.listdir(os.path.join(img_dir_out_num, folder)))
    num = 10 - num
    for pos in range(len(imgs_list) - 5, len(imgs_list)):
        link = path +  '\\' +  imgs_list[pos]
        out = path_out +  '\\' +  imgs_list[pos]
        try:
            frame = cv2.imread(link)
            ret = model.get_input(frame)
            if ret is not None:
                shutil.copy(link, out)  
                sum += 1
                print(sum)
                # if sum >= 5:
                #     break  
            
        except:
            pass

