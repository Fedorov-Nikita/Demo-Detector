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
	
def plot_predictions(numpy_img, preds, tresh=0.5):
	boxes = preds['boxes'][preds['scores'] > tresh].detach().numpy()
	labels = preds['labels'][preds['scores'] > tresh].detach().numpy()
	for i, box in enumerate(boxes):
		numpy_img = cv2.rectangle(numpy_img.astype('float32'), (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), (255, 0, 0), 3)
		# coco_labels[labels[i]]
		numpy_img = cv2.putText(numpy_img, 
			coco_labels[labels[i]-1], 
			(
				int(box[0]),
				int(box[1])-5
			), 
			cv2.FONT_HERSHEY_SIMPLEX,
			fontScale = 2,
			color = 255,
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
		st.image(image_data)
		img_numpy = np.array(Image.open(io.BytesIO(image_data)))
		return img_numpy

def main():
	st.title("Time to detect what you want")
	img = load_image()
	if img is not None:
		threshold = st.sidebar.slider("Set the threshold for the detector", 0., 1.)
		model = load_model()
		clicked = st.sidebar.button('Detect objects')
		if clicked:
			detect_my_image(img, model, threshold)
			st.success('Done')
			st.snow()
	
if __name__ == '__main__':
	main()