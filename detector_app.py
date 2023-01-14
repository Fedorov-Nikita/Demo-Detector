import io
from PIL import Image

import streamlit as st
import cv2
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
# import pandas as pd
# import time

coco_labels= ['person', 'bicycle', 'car', 'motorcycle', 'airplane',
'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
'hat', 'backpack', 'umbrella','shoe','eye glasses','handbag','tie',
'suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat',
'baseball glove','skateboard','surfboard','tennis racket','bottle',
'plate','wine glass','cup','fork','knife','spoon','bowl','banana',
'apple','sandwich','orange','broccoli','carrot','hot dog','pizza',
'donut','cake','chair','couch','potted plant','bed','mirror','dining table',
'window','desk','toilet','door','tv','laptop','mouse','remote','keyboard',
'cell phone','microwave','oven','toaster','sink','refrigerator','blender',
'book','clock','vase','scissors','teddy bear','hair drier','toothbrush',
'hair brush']

@st.cache(allow_output_mutation=True)
def load_model():
	# load a model pre-trained pre-trained on COCO
	model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
	model = model.eval()
	return model
	
def plot_predictions(numpy_img, preds, tresh):
	boxes = preds['boxes'][preds['scores'] > tresh].detach().numpy()
	labels = preds['labels'][preds['scores'] > tresh].detach().numpy()
	scores = preds['scores'][preds['scores'] > tresh].detach().numpy()
	img_y = numpy_img.shape[0]
	img_x = numpy_img.shape[1]
	for i, box in enumerate(boxes):
		np.random.seed(labels[i]-1)
		color_ = (np.random.randint(0, 255),np.random.randint(0, 255),np.random.randint(0, 255))
		label = f'{coco_labels[labels[i]-1].capitalize()}: {round(scores[i]*100, 2)}%'
		(w, h), _ = cv2.getTextSize(
			label, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
		numpy_img = cv2.rectangle(numpy_img, (int(box[0]), int(box[1])), (int(box[2]),int(box[3])), color=color_, thickness=3)
		# coco_labels[labels[i]]
		x_lu = int(box[0])
		x_rd = int(box[0] + w)
		y_lu = int(box[1] - h*1.2)
		y_rd = int(box[1])
		d_x = 0
		d_y = 0
		if (x_lu < 0):
			d_x = 0 - x_lu
		if (y_lu < 0):
			d_y = 0 - y_lu
		if (x_rd > img_x):
			d_x = img_x - x_rd
		if (y_rd > img_y):
			d_y = img_y - y_rd
		x_lu = x_lu + d_x
		x_rd = x_rd + d_x
		y_lu = y_lu + d_y
		y_rd = y_rd + d_y
		numpy_img = cv2.rectangle(numpy_img, (x_lu, y_lu), (x_rd, y_rd), color_, -1)
		# st.write(int(box[0]), int(box[1] - h*1.2),'---', int(box[0] + w), int(box[1]))
		numpy_img = cv2.putText(numpy_img, 
			label, 
			(
				x_lu,
				y_rd-5
			), 
			cv2.FONT_HERSHEY_SIMPLEX,
			fontScale = 2,
			color = (255,255,255),
			thickness = 3
		)
	numpy_img = numpy_img.astype('uint')
	st.image(numpy_img)

def detect_my_image(img_numpy, model, tresh):
	img = torch.from_numpy(img_numpy.astype('float32')).permute(2,0,1)
	img = img / 255.
	predictions = model(img[None,...])
	plot_predictions(img_numpy, predictions[0], tresh)

def load_image():
	uploaded_file = st.sidebar.file_uploader(label='Upload a photo')
	if uploaded_file is not None:
		image_data = uploaded_file.getvalue()
		img_numpy = np.array(Image.open(io.BytesIO(image_data)))
		return img_numpy

def main():
	st.title("Time to detect what you want")
	img = load_image()
	if img is not None:
		threshold = st.sidebar.slider("Set the threshold for the detector", 0., 1.)
		if threshold == .0:
			threshold = 0.75
		model = load_model()
		clicked = st.sidebar.button('Detect objects')
		if clicked:
			with st.spinner('Please wait for us to carry out a detection...'):
				detect_my_image(img, model, threshold)
	
if __name__ == '__main__':
	main()