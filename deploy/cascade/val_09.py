# val mask


import argparse
from tokenize import tabsize

import numpy as np
import cv2
from face_recognizer import FaceRecognizer
from imutils.video import VideoStream
import os

#https://viblo.asia/p/viet-cli-trong-python-de-dang-voi-argparse-XL6lA2ar5ek
parser = argparse.ArgumentParser(description='face_recognization')#Tạo  biến parser .phần ArgumentParser sẽ trả về một object, hay một đối tượng. Từ bước này trở đi biến parser sẽ lưu giữ các thông tin cần thiết để truyền các biến từ CLI vào chương trình Python
# Tham số description được sử dụng để cung cấp thông tin mô tả chương trình
parser.add_argument('--face_db_root', type=str, default=r'D:\TruongCongHau\KhoaLuanTotNghiep_2021\khoaluantotnghiep\deploy\images\du_lieu_sinh_vien\374_09\mask', help='the root path of target database')#Thêm tham số face_db_root
parser.add_argument('--input_video_path', type=str, default='D:/mask_data/02.mp4', help='the path of input video')
parser.add_argument('--output_video_path', type=str, default='output.mp4', help='the path of input video')
#add_argument định nghĩa cách mà các biến từ CLI sẽ được truyền vào. Mỗi lần gọi add_argument sẽ xử lý một tham số duy nhất

args = parser.parse_args()
#biến các tham số được gửi vào từ CLI thành các thuộc tính của 1 object và trả về object đó.

# Tạo file embed
# recognizer = FaceRecognizer()
# recognizer.create_known_faces(args.face_db_root)
#recognizer.test_100x()

recognizer = FaceRecognizer()
# Load file train
data = np.load('embed_test.npz')
# recognizer = FaceRecognizer()
recognizer.create_known_faces_read(data['arr_0'], data['arr_1'])



# recognizer.create_known_faces(args.face_db_root)
# 
input_dir_test = 'D:/TruongCongHau/KhoaLuanTotNghiep_2021/khoaluantotnghiep/deploy/images/du_lieu_sinh_vien/374_09/test_mask'
folders = os.listdir(input_dir_test)  # data

sum = 0
count_right = 0

for folder in folders:
    label = folder
    imgs = os.listdir(os.path.join(input_dir_test, folder))
    sum += len(imgs)
    for i in range(len(imgs)):
         link = input_dir_test + '\\' + folder + '\\' + imgs[i]
         image = cv2.imread(link)
         item = recognizer.recognize(image,0)
         if item is None:
              sum -=1
         if item:
             name, (left, top, right, bottom), _, score = item
             print(name, ' - ', label)
             if(name == folder):
                  count_right += 1
print('tong anh ', sum)
print('anh doan dung', count_right)
print('Acc_test', count_right / float(sum))


