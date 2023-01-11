import streamlit as st
# import numpy as np
# import pandas as pd
# import time



# uploaded_file = st.sidebar.file_uploader("Upload a photo")
# uploaded_file = st.sidebar.text_input("Or insert a link to the image")

def load_image():
	uploaded_file = st.sidebar.file_uploader(label='Upload a photo')
	# uploaded_file = st.text_input("Or insert a link to the image")
	if uploaded_file is not None:
		image_data = uploaded_file.getvalue()
		st.image(image_data)
		set_detect_params()

def set_detect_params():
	threshold = st.sidebar.slider("Set the threshold for the detector", 0., 1.)
	clicked = st.sidebar.button('Detect objects')
# with st.spinner('Please wait...'):
# 	time.sleep(5)
# 	# st.success('Done')
# 	st.snow()

def main():
	st.title("Time to detect what you want")
	load_image()
	
if __name__ == '__main__':
	main()