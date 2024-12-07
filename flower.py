import os
import tensorflow as tf
import numpy as np
from keras.models import load_model
import streamlit as st

# Streamlit App Header
st.header('🌸 Flower Classification using CNN 🌼')

# Flower categories (update this according to your model's classes)
flower_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Predefined path for the pretrained Keras model
model_path = r'C:\Users\Lenovo\OneDrive\Desktop\Flower_classification\Flower_Recog_Model.keras'

# Load the pretrained model
try:
    model = load_model(model_path)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Classification function
def classify_images(image):
    # Resize the image to the model's input size
    input_shape = model.input_shape[1:3]  # Get model's expected input size (height, width)
    input_image = tf.image.resize(image, input_shape)  # Resize image
    input_image = tf.expand_dims(input_image, 0)  # Add batch dimension
    predictions = model.predict(input_image)
    result = tf.nn.softmax(predictions[0])  # Apply softmax for classification
    class_index = np.argmax(result)  # Get the index of the highest confidence class
    confidence = np.max(result) * 100  # Get the confidence score
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
