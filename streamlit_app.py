import streamlit as st
import streamlit_ext as ste
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import tensorflow as tf
import torch
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.models import Sequential, Model, load_model
import io
from base64 import encodebytes
from PIL import Image
from IPython.display import display
from PIL import Image
import time
from io import BytesIO

# Load YOLOv5 model (PyTorch)
@st.cache(allow_output_mutation=True)
def load_yolo_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
    model.eval()
    return model

# Load DenseNet model
@st.cache(allow_output_mutation=True)
def load_densenet_model():
    model = load_model('densenet_stage1_all-0.954.hdf5')
    return model

# Prediction function
def predict_damaged(image, yolo_model, densenet_model):
    # Load and preprocess the image
    img_array = img_to_array(image)
    img_array = img_array.reshape((1,) + img_array.shape)
    
    # Check if damaged using DenseNet
    prediction = densenet_model.predict(img_array)
    if prediction <= 0.5:
        return "Not Damaged"

    # Use YOLOv5 to detect the type and severity of damage
    results = yolo_model(image)
    return results

# Main app
def main():
    st.title('CAR DAMAGE ASSESSMENT')      
    st.image('car_assess_head.jfif') 

    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Load models
        with st.spinner("Loading models..."):
            yolo_model = load_yolo_model()
            densenet_model = load_densenet_model()

        # Make predictions
        with st.spinner("Predicting..."):
            prediction = predict_damaged(image, yolo_model, densenet_model)

        if prediction == "Not Damaged":
            st.warning("Please check your image! The vehicle appears to be undamaged.")
        else:
            st.success("Vehicle is damaged. Damage detected:")
            st.image(prediction.render()[0], caption='Damage Detection')

            # Download button for the image
            st.download_button("Download image", prediction.render()[0], "output.jpg")

if __name__ == "__main__":
    main()
