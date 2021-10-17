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
parser.add_argument('--model', default='../../recognition/models/10_use/model,1', help='path to load model.')
parser.add_argument('--haarcasecade', default='../haarcascade_frontalface_default.xml', help='path to load haarcasecade.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()
model = face_model.FaceModel(args)

with open ('../data/10_use/list_feature_mtcnn_lfw.p', 'rb') as fp: 
    list_feature = pickle.load(fp)
with open ('../data/10_use/name_label_mtcnn_lfw.p', 'rb') as fp:   
    list_label = pickle.load(fp)

color=(255, 255, 0)
bcolor=(0,0,255)
thickness = 1
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1

# print('*'*8, 'into video')
# root_video = "../image-test/2020_08_09_13_12_IMG_2999.MOV"
VIDEO_STREAM_OUT = "../image-test/abc_cascade_hau.avi"
# cap = cv2.VideoCapture('root_video')
writer = None
# cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
# print('get video')
i = 0

#sd camera realtime
input_movie = VideoStream(src=0).start()

# kích thước video
video_size = (1024, 720)


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    

    # ret, frame = cap.read()
    if not ret:
        break
    if i%1 == 0:
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
                    # acc = 1 - x
                else:
                    name = "Unknow!"
                acc = (len(list_dist)/10)*100
                # acc = (len(list_dist)/len(list_dist))*100
                # acc = len(list_dist)
                box=bbox_unit.astype(np.int).flatten()
                cv2.rectangle(frame, (box[0],box[1]), (box[2],box[3]), bcolor,2)
                cv2.putText(frame, str(acc)+"%", (box[0],box[1]), font, fontScale, color, thickness, cv2.LINE_AA)
                cv2.putText(frame, name, (box[0],box[3]), font, fontScale, color, thickness, cv2.LINE_AA)
        except Exception as e:
            frame = frame
    cv2.imshow('My camera...',frame)    
    # if writer is None:
    #     fourcc = cv2.VideoWriter_fourcc(*"XVID")
    #     writer = cv2.VideoWriter(VIDEO_STREAM_OUT, fourcc, 20,(frame.shape[1], frame.shape[0]), True)
    # writer.write(frame)

    i+=1
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
# writer.release()
cv2.destroyAllWindows()
print("SuccessFully!")