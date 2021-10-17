import opencv_model
import argparse
import cv2
import os
import sys
import time
import numpy as np


parser = argparse.ArgumentParser(description='face model test')
#general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
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
        self.model = ''
        self.ga_model = ''
        self.haarcasecade = './haarcascade_frontalface_default.xml'
        self.threshold = 1.24
        self.flip = 0
        self.det = 0

args = Args()
model = opencv_model.FaceModel(args)
print("Bat dau cat anh")

imgs_dir = './images'
faces_save_dst = 'val/99sv_cascade/'

if not os.path.exists(faces_save_dst):
    os.mkdir(faces_save_dst)
folders = os.listdir(imgs_dir)
start = time.time()
for folder in folders:
    imgs = os.listdir(os.path.join(imgs_dir,folder))
    for file in imgs:
        print(file)
    cnt = 1
    for img in imgs:
        img_root_path = os.path.join(imgs_dir,folder,img)
        try:
            pic = cv2.imread(img_root_path)
            pic,bbox,landmark = model.get_input_v2(pic)
            if type(pic) == np.ndarray:
                pic = np.transpose(pic,(1,2,0))[:, :, ::-1]
                if not os.path.exists(os.path.join(faces_save_dst,folder)):
                    os.mkdir(os.path.join(faces_save_dst,folder))
                if cnt == 10:
                    cv2.imwrite(os.path.join(faces_save_dst,folder,folder + '_00' + str(cnt) + '.JPG'),pic)
                else:
                    cv2.imwrite(os.path.join(faces_save_dst,folder,folder + '_000' + str(cnt) + '.JPG'),pic)
                cnt+=1                   
        except Exception as e:
            print('Error occurred : ' + str(e))
end = time.time()
interval = end - start

print("Total time: %s"%str(interval))
print("SuccessFully!")

