from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
#import tensorflow as tf
import numpy as np
import mxnet as mx
import random
import cv2
import sklearn
from sklearn.decomposition import PCA
from time import sleep
from easydict import EasyDict as edict
from opencv_detector import OpenCV_Detector
sys.path.append(os.path.join(os.path.dirname(__file__), '..','src','common'))
import face_image
import face_preprocess


def get_model(ctx, image_size, model_str, layer):
	_vec = model_str.split(',')
	assert len(_vec)==2
	prefix = _vec[0]
	epoch = int(_vec[1])
	print('loading',prefix, epoch)
	sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
	all_layers = sym.get_internals()
	sym = all_layers[layer+'_output']
	model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
	model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
	model.set_params(arg_params, aux_params)
	return model

class FaceModel:
	def __init__(self, args):
		self.args = args
		ctx = mx.cpu()
		_vec = args.image_size.split(',')
		assert len(_vec)==2
		image_size = (int(_vec[0]), int(_vec[1]))
		self.model = None
		if len(args.model) > 0:
			self.model = get_model(ctx, image_size, args.model, 'fc1')

		self.threshold = args.threshold
		self.det_threshold = [0.6,0.7,0.8]
		self.image_size = image_size

		mtcnn_path = os.path.join(os.path.dirname(__file__), 'mtcnn-model')

		if args.det==0:
			detector = OpenCV_Detector(model_folder=mtcnn_path, ctx=ctx, threshold=self.det_threshold)
		else:
			detector = OpenCV_Detector(model_folder=mtcnn_path, ctx=ctx, threshold=[0.0,0.0,0.2])

		self.detector = detector

	def get_input(self, face_img):
		ret = self.detector.detect_face(face_img, path = self.args.haarcasecade)
		if ret is None:
			return None
		bbox, points = ret
		aligned_list = []
		bbox_list = []
		points_list = []
		if (bbox.shape[0] == 0):
			return None
		for i in range(bbox.shape[0]):
			bbox_ = bbox[i,0:4]
			bbox_list.append(bbox_)
			points_ = points[i,:].reshape((2,5)).T
			points_list.append(points_)
			nimg = face_preprocess.preprocess(face_img, bbox_, points_, image_size='112,112')
			nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
			aligned = np.transpose(nimg, (2,0,1))
			aligned_list.append(aligned)
		return aligned_list,bbox_list,points_list

	def get_input_v2(self, face_img):
		ret = self.detector.detect_face(face_img, path = self.args.haarcasecade)
		if ret is None:
			return None
		bbox, points = ret
		bbox = bbox[0,0:4]
		points = points[0,:].reshape((2,5)).T
		nimg = face_preprocess.preprocess(face_img, bbox, points, image_size='112,112')
		nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
		aligned = np.transpose(nimg, (2,0,1))
		return aligned,bbox,points

	def get_feature(self, aligned):
		input_blob = np.expand_dims(aligned, axis=0)
		data = mx.nd.array(input_blob)
		db = mx.io.DataBatch(data=(data,))
		self.model.forward(db, is_train=False)
		embedding = self.model.get_outputs()[0].asnumpy()
		embedding = sklearn.preprocessing.normalize(embedding).flatten()
		return embedding