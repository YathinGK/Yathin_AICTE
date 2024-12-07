import os
import tensorflow as tf
import numpy as np
from keras.models import load_model
import streamlit as st

# Streamlit App Header
st.header(' Flower Classification using CNN ')

# Flower categories
flower_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Load the trained model
model_path = r'Flower_Recog_Model.h5'
model = load_model(model_path)

# Classification function
def classify_images(image):
    input_image = tf.image.resize(image, (180, 180))  # Resize to model's input size
    input_image = tf.expand_dims(input_image, 0)  # Add batch dimension
    predictions = model.predict(input_image)
    result = tf.nn.softmax(predictions[0])
    class_index = np.argmax(result)
    confidence = np.max(result) * 100
    return f"The image is classified as **{flower_names[class_index]}** with a confidence score of **{confidence:.2f}%**."

# File uploader for image
uploaded_file = st.file_uploader('Upload an Image of a Flower (JPEG/PNG)', type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Load and preprocess the image
    file_bytes = uploaded_file.read()
    input_image = tf.image.decode_image(file_bytes, channels=3)

    # Display classification result
    result = classify_images(input_image)
    st.write(result)
