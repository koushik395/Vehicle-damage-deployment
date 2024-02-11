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

#densenet models
@st.cache_resource(show_spinner=False)  # Cache the model so we don't reload it everytime
def load_models():
    model1 = load_model('densenet_stage1_all-0.954.hdf5')
    
    return model1

# YOLOv5 model (PyTorch)
@st.cache_resource(show_spinner=False)
def load_yolo_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
    model.eval()
    return model

#Creating a single function which predicts wether Car is damaged or not and localizing the damage, severity of damage for a single img
def report(img_path,model,model1):
    report_pred = []

    # img = load_img(os.getcwd()+f'/{img_path}',target_size = (256,256))
    #Converting into array
    img_arr = img_to_array(img_path)
    img_arr = img_arr.reshape((1,) + img_arr.shape)

    #Checking if Damaged or not
    s1_pred = model1.predict(img_arr)
    if s1_pred <=0.5:
        report_pred.append('Damaged')
    else:
        report_pred.append('Not Damaged')
        return report_pred

    #Using YOLO to detect the damage type
    results = model(img_path)
    results.render()
    Image.fromarray(results.ims[0]).save("static/images/pred.jpg")

    return report_pred

        
# Main app
def main():
    app_mode = st.sidebar.selectbox('Select Page',['Home','Prediction']) #two pages      

    if app_mode=='Home':    
        st.title('CAR DAMAGE ASSESSMENT')      
        st.image('car_assess_head.jfif') 

    elif app_mode == 'Prediction':       
        st.subheader('PLEASE UPLOAD YOUR DAMAGED CAR')
        uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            with st.status("Downloading data...", expanded=True) as status:
                st.write("Loading....!")
                time.sleep(2)
                st.write("Please wait....!")
                time.sleep(1)
                st.write("About to complete....!")
                time.sleep(1)
                model1 = load_models()
                model = load_yolo_model()
                status.update(label="Models Loaded", state="complete", expanded=False)  # Display a spinner while loading models

            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            image = image.resize((256, 256))

            with st.spinner("Predicting..."):  # Display a spinner while making predictions
                preds = report(image,model,model1)

            st.balloons()
            if preds[0] == 'Damaged':
                pred_img = Image.open("static/images/pred.jpg")
                st.success("Output generated successfully")
                st.subheader("The vehicle is damaged.")
                st.subheader("Below is the type of damage detected.")
                st.image(pred_img, caption='Damage Detection')

                buf = BytesIO()
                pred_img.save(buf, format="JPEG")
                byte_im = buf.getvalue()
                ste.download_button("Download image", byte_im, "output.jpeg")
            else:
                st.warning("Please check your image!")
                st.subheader("Are you sure the vehicle is damaged?. Please check once.")


if __name__ == "__main__":
    main()
