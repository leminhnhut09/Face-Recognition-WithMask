import cv2
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
import opencv_model as face_model
import argparse
import numpy as np
import pickle
import imutils
from imutils.video import VideoStream
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='../recognition/models/10_new(haft)/model,1', help='path to load model.')
parser.add_argument('--haarcasecade', default='../haarcascade_frontalface_default.xml', help='path to load haarcasecade.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')


args = parser.parse_args()
model = face_model.FaceModel(args)

def detect_and_predict_mask(frame, faceNet, maskNet):
     	# grab the dimensions of the frame and then construct a blob
	# from it
	frame = imutils.resize(frame, width=400)
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

# load our serialized face detector model from disk
prototxtPath = r"D:\TruongCongHau\KhoaLuanTotNghiep_2021\khoaluantotnghiep\deploy\cascade\face_detector\deploy.prototxt"
weightsPath = r"D:\TruongCongHau\KhoaLuanTotNghiep_2021\khoaluantotnghiep\deploy\cascade\face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model(r"D:\TruongCongHau\KhoaLuanTotNghiep_2021\khoaluantotnghiep\deploy\cascade\mask_detector.model")


with open (r'D:\TruongCongHau\KhoaLuanTotNghiep_2021\khoaluantotnghiep\deploy\data\10_new(haft)\list_feature_mtcnn_lfw.p', 'rb') as fp: 
    list_feature = pickle.load(fp)
with open (r'D:\TruongCongHau\KhoaLuanTotNghiep_2021\khoaluantotnghiep\deploy\data\10_new(haft)\name_label_mtcnn_lfw.p', 'rb') as fp:   
    list_label = pickle.load(fp)

    
print(list_feature[0].shape)


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
    frame = imutils.resize(frame, width=400)
    # ret, frame = cap.read()
    if not ret:
        break
    if i%1 == 0:
     #    try:
            (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
            for (box, pred) in zip(locs, preds):
                    (startX, startY, endX, endY) = box
                    (mask, withoutMask) = pred
                    if withoutMask > mask:
                    # if mask > withoutMask:
                        height = int((box[3] - box[1]) / 2) + box[1]
                        img_crop = frame[box[1]:height, box[0]:box[2], :]


                        # img_crop = frame[box[1]:box[3], box[0]:box[2], :]


                        img_crop = cv2.resize(img_crop, (112, 112))
                        img_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
                        # plt.imshow(img_crop)
                        # plt.show()
                        img_crop = np.transpose(img_crop, (2,0,1))
                        
                      
                        # print(img_crop.shape)
                        f = model.get_feature(img_crop)
                        print(f.shape)
                    #     print(f)
                        bien = False
                        list_dist = []
                        list_i = []
                        for i in range(len(list_feature)):
                             dist = np.sum(np.square(list_feature[i]-f))
                         #     print(list_feature[i])
                             print(dist)
                             if dist < 2.1:
                                 
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
                    #     cv2.rectangle(frame, (box[0],box[1]), (box[2],box[3]), bcolor,2)
                    
                        # frame = cv2.rectangle(frame, (box[0],height), (box[2],box[3]), bcolor,2)
                        frame = cv2.rectangle(frame, (box[0],box[1]), (box[2],box[3]), bcolor,2)
                        cv2.putText(frame, str(acc)+"%", (box[0],box[1]), font, fontScale, color, thickness, cv2.LINE_AA)
                        cv2.putText(frame, name, (box[0],box[3]), font, fontScale, color, thickness, cv2.LINE_AA)           
     #    except Exception as e:
     #        frame = frame
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