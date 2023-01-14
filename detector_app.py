import io
from PIL import Image
import base64
import uuid
import re

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
		x_lu = int(box[0])
		x_rd = int(box[0] + w)
		y_lu = int(box[1] - h*1.2)
		y_rd = int(box[1])
		# Correct the coordinates going beyond the edge of the image
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
		# Draw labels
		numpy_img = cv2.rectangle(numpy_img, (x_lu, y_lu), (x_rd, y_rd), color_, -1)
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
	result = Image.fromarray(numpy_img)
	numpy_img = numpy_img.astype('uint')
	st.image(numpy_img)
	if numpy_img.shape[2] > 3:
		isPNG = True
		output_extension = ".png"
	else:
		isPNG = False
		output_extension = ".jpg"
	st.sidebar.markdown(download_button(result, f"detected_photo{output_extension}", "Download detected photo", isPNG), unsafe_allow_html=True)

def download_button(object_to_download, download_filename, button_text, isPNG):
	"""
	Generates a link to download the given object_to_download.

	Params:
	------
	object_to_download:  The object to be downloaded.
	download_filename (str): filename and extension of file. e.g. mydata.csv,
	some_txt_output.txt download_link_text (str): Text to display for download
	link.
	button_text (str): Text to display on download button (e.g. 'click here to download file')
	pickle_it (bool): If True, pickle file.

	Returns:
	-------
	(str): the anchor tag to download object_to_download

	Examples:
	--------
	download_link(Pillow_image_from_cv_matrix, 'your_image.jpg', 'Click to me to download!')
	"""

	buffered = io.BytesIO()
	if isPNG:
		object_to_download.save(buffered, format="PNG")
	else:
		object_to_download.save(buffered, format="JPEG")
	b64 = base64.b64encode(buffered.getvalue()).decode()

	button_uuid = str(uuid.uuid4()).replace('-', '')
	button_id = re.sub('\d+', '', button_uuid)

	custom_css = f""" 
		<style>
			#{button_id} {{
				display: inline-flex;
				align-items: center;
				justify-content: center;
				background-color: rgb(255, 255, 255);
				color: rgb(38, 39, 48);
				padding: .25rem .75rem;
				position: relative;
				text-decoration: none;
				border-radius: 4px;
				border-width: 1px;
				border-style: solid;
				border-color: rgb(230, 234, 241);
				border-image: initial;
			}} 

			#{button_id}:hover {{
				border-color: rgb(246, 51, 102);
				color: rgb(246, 51, 102);
			}}
			#{button_id}:active {{
				box-shadow: none;
				background-color: rgb(246, 51, 102);
				color: white;
				}}
		</style> """

	dl_link = custom_css + f'<a download="{download_filename}" id="{button_id}" href="data:file/txt;base64,{b64}">{button_text}</a><br></br>'
	return dl_link
	

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
	st.title("It's time to detect what's in your photo")
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