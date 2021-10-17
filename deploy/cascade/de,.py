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
import random

# đường dẫn lấy ảnh
# img_dir_in_mask = r'C:\Users\CongHau\Desktop\Data_GuiThayHa\DATA_374\mask'
# img_dir_in_nomask = r'C:\Users\CongHau\Desktop\Data_GuiThayHa\DATA_374\nomask'


# img_dir_out_train = r'C:\Users\CongHau\Desktop\Data_GuiThayHa\DATA_374\data_chinh\08\train'
# img_dir_out_test = r'C:\Users\CongHau\Desktop\Data_GuiThayHa\DATA_374\data_chinh\08\test'


# folders = os.listdir(img_dir_in_mask)  # data

# for folder in folders:  # anh
#     imgs_list_mask = os.listdir(os.path.join(img_dir_in_mask, folder))  # lst img
#     imgs_list_nomask = os.listdir(os.path.join(img_dir_in_nomask, folder))  # lst img
#     random.shuffle(imgs_list_mask)
#     random.shuffle(imgs_list_nomask)
#     os.makedirs(img_dir_out_train + '/' + folder)
#     os.makedirs(img_dir_out_test + '/' + folder)
#     heso = 0.8
    

#     #  chia mask
#     if(len(imgs_list_mask) == 1):
#         img_in = img_dir_in_mask + '/' + folder + '/' + imgs_list_mask[0]
#         img_out = img_dir_out_train + '/' + folder + '/' + imgs_list_mask[0]
#         shutil.copy(img_in, img_out)
#     # > 1
#     else:
#         num_move_train = int(heso * len(imgs_list_mask))
#         for idx in range(len(imgs_list_mask)):
#             img_in = img_dir_in_mask + '/' + folder + '/' + imgs_list_mask[idx]
#             img_out_train = img_dir_out_train + '/' + folder + '/' + imgs_list_mask[idx]
#             img_out_test = img_dir_out_test + '/' + folder + '/' + imgs_list_mask[idx]
#             if num_move_train == 0:
#                  shutil.copy(img_in, img_out_test)
#                  continue
#             shutil.copy(img_in, img_out_train)
#             num_move_train -= 1
#     #  chia nomask
#     imgs_list_train = os.listdir(os.path.join(img_dir_out_train, folder))  # lst img
#     imgs_list_test = os.listdir(os.path.join(img_dir_out_test, folder))  # lst img
#     num_train = 8 - len(imgs_list_train)
#     num_test = 2 - len(imgs_list_test)
#     for pos in range(len(imgs_list_nomask)):
#         img_in = img_dir_in_nomask + '/' + folder + '/' + imgs_list_nomask[pos]
#         img_out_train = img_dir_out_train + '/' + folder + '/' + imgs_list_nomask[pos]
#         img_out_test = img_dir_out_test + '/' + folder + '/' + imgs_list_nomask[pos]
#         if num_train == 0:
#             shutil.copy(img_in, img_out_test)
#             continue
#         shutil.copy(img_in, img_out_train)
#         num_train -= 1
        
        















# # đường dẫn lấy ảnh
img_dir_out = r'C:\Users\CongHau\Desktop\Data_GuiThayHa\DATA_374\data_chinh\08\test'
img_dir_out_num = r'C:\Users\CongHau\Desktop\Data_GuiThayHa\DATA_374\data_chinh\08\train'
folders = os.listdir(img_dir_out)  # data
sum = 0
f = 0

for folder in folders:  # anh
    imgs_list1 = os.listdir(os.path.join(img_dir_out, folder))  # lst img
    imgs_list2 = os.listdir(os.path.join(img_dir_out_num, folder))  # lst img
    num = len(imgs_list1) + len(imgs_list2)
    sum += num
    if num < 10:
        print(folder,'-', num)
     #   f += 1
print('tong', sum)
print('l', f)


