import os
import tensorflow as tf
import numpy as np
from keras.models import load_model
import streamlit as st


st.header('Flower Classification')


flower_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']


model_filename = "Flower_Recog_Model.keras"
current_dir = os.path.dirname(os.path.abspath(__file__))  
model_path = os.path.join(current_dir, model_filename)

if os.path.exists(model_path):

    model = load_model(model_path)  
    st.write("Model loaded successfully!")
else:
    st.error(f"Model file '{model_filename}' not found in {current_dir}. Please ensure the file is placed correctly.")

# Classification function
def classify_images(image):
    input_image = tf.image.resize(image, (180, 180))  
    input_image = tf.expand_dims(input_image, 0)  
    predictions = model.predict(input_image)
    result = tf.nn.softmax(predictions[0])
    class_index = np.argmax(result)
    confidence = np.max(result) * 100
    return f"The image is classified as **{flower_names[class_index]}** with a confidence score of **{confidence:.2f}%**."


uploaded_file = st.file_uploader('Upload an Image of a Flower (JPEG/PNG)', type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    
    file_bytes = uploaded_file.read()
    input_image = tf.image.decode_image(file_bytes, channels=3)

    
    result = classify_images(input_image)
    st.write(result)
