import face_model
import argparse
import cv2
import sys
import os
import pickle
import time
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
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


# load our serialized face detector model from disk
prototxtPath = r"D:\TruongCongHau\KhoaLuanTotNghiep_2021\khoaluantotnghiep\deploy\cascade\face_detector\deploy.prototxt"
weightsPath = r"D:\TruongCongHau\KhoaLuanTotNghiep_2021\khoaluantotnghiep\deploy\cascade\face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model(r"D:\TruongCongHau\KhoaLuanTotNghiep_2021\khoaluantotnghiep\deploy\cascade\mask_detector.model")


def detect_and_predict_mask(frame, faceNet, maskNet):
         # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)



class Args():
    def __init__(self):
        self.image_size = '112,112'
        self.gpu = 0
        self.model = '../recognition/models/10_mask/model,0001'#đường dẫn đến model vừa tạo
        self.ga_model = ''  #lay tuổi + gt
        self.threshold = 1.24
        self.flip = 0
        self.det = 0

args = Args()
model = face_model.FaceModel(args)
imgs_dir = './images/du_lieu_sinh_vien/10_new(haft)' # đường dẫn hình
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
                # img,bb,lm = model.get_input(img)
                (locs, preds) = detect_and_predict_mask(img, faceNet, maskNet)
                for (box, pred) in zip(locs, preds):
                    (startX, startY, endX, endY) = box
                    (mask, withoutMask) = pred
                    

                    # if withoutMask > mask:
                    if mask > withoutMask:
                             
                        height = int((box[3] - box[1]) / 2) + box[1]
                        img_crop = img[box[1]:height, box[0]:box[2], :]
                        
                        # img_crop = img[box[1]:box[3], box[0]:box[2], :]

                        img_crop = cv2.resize(img_crop, (112, 112))
                        img_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
                        # plt.imshow(img_crop)
                        # plt.show()
                        img_crop = np.transpose(img_crop, (2,0,1))
                        
                        # print(img_crop.shape)
                        img = model.get_feature(img_crop)
                        list_feature.append(img)
                        # print(list_feature[0].shape)
                        name_label.append(folder)
                        bien = True
            except Exception as e:
                print('Error occurred : ' + str(e))
        else:
            print("Ko doc duoc hinh anh: ",img_root_path)
        if bien == True:
            list_len.append(len(imgs))
            
with open('./images/du_lieu_sinh_vien/10_new(haft)/list_len_mtcnn_lfw.p', 'wb') as fp: #list lưu số lượng hình của 1 thư mục
    pickle.dump(list_len, fp)
    
with open('./images/du_lieu_sinh_vien/10_new(haft)/list_feature_mtcnn_lfw.p', 'wb') as fp: #vị trí lưu listfeature
    pickle.dump(list_feature, fp)
    
with open('./images/du_lieu_sinh_vien/10_new(haft)/name_label_mtcnn_lfw.p', 'wb') as fp: #list lưu tên thư mục
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