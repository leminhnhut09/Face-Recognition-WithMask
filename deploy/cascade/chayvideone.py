import cv2
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
import opencv_model as face_model
import argparse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from face_recognizer import FaceRecognizer
import imutils
import pickle
from imutils.video import VideoStream
parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='../../recognition/models/16/model,0', help='path to load model.')
parser.add_argument('--haarcasecade', default='../haarcascade_frontalface_default.xml', help='path to load haarcasecade.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')

# arg mask
parser.add_argument('--face_db_root', type=str, default='data/mask_nomask', help='the root path of target database')#Thêm tham số face_db_root
parser.add_argument('--input_video_path', type=str, default='D:/mask_data/02.mp4', help='the path of input video')
parser.add_argument('--output_video_path', type=str, default='output.mp4', help='the path of input video')


args = parser.parse_args()
model = face_model.FaceModel(args)

recognizer = FaceRecognizer()
recognizer.create_known_faces(args.face_db_root)

with open ('../data/mtcnn_16/list_feature_mtcnn_lfw.p', 'rb') as fp: 
    list_feature = pickle.load(fp)
with open ('../data/mtcnn_16/name_label_mtcnn_lfw.p', 'rb') as fp:   
    list_label = pickle.load(fp)

color=(255, 255, 0)
bcolor=(0,0,255)
thickness = 1
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1


# mask
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
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")


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
#     frame = imutils.resize(frame, width=400)
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred
        

    if mask < withoutMask:



    # ret, frame = cap.read()
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
                else:
                    name = "Unknow!"
                acc = (len(list_dist)/10)*100
                box=bbox_unit.astype(np.int).flatten()
                cv2.rectangle(frame, (box[0],box[1]), (box[2],box[3]), bcolor,2)
                cv2.putText(frame, str(acc)+"%", (box[0],box[1]), font, fontScale, color, thickness, cv2.LINE_AA)
                cv2.putText(frame, name, (box[0],box[3]), font, fontScale, color, thickness, cv2.LINE_AA)
        except Exception as e:
            frame = frame
    else:
        item = recognizer.recognize(frame, 0.5)
        if item:
            name, (left, top, right, bottom), _, score = item
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, "%s %.3f" % (name, score), (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)


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