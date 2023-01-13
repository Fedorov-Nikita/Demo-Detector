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

@st.cache(allow_output_mutation=True)
def load_model():
	# load a model pre-trained pre-trained on COCO
	model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
	model = model.eval()
	return model
	
def plot_predictions(numpy_img, preds, tresh=0.5):
	boxes = preds['boxes'][preds['scores'] > tresh].detach().numpy()
	for box in boxes:
		numpy_img = cv2.rectangle(numpy_img.astype('float32'), (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), (255, 0, 0), 3)
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