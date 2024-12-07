import os
import tensorflow as tf
import numpy as np
from keras.models import load_model
import streamlit as st

# Streamlit App Header
st.header('🌼 Flower Classification using CNN')

# Define flower categories
flower_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Load the trained model safely
def load_trained_model(model_path):
    try:
        model = load_model(model_path)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Failed to load the model. Error: {e}")
        return None

# Path to the model
model_path = 'Flower_Recog_Model.keras'
model = load_trained_model(model_path)

# Function to classify the uploaded image
def classify_image(image, model):
    try:
        # Preprocess the image
        input_image = tf.image.resize(image, (180, 180))  # Resize to model's expected input size
        input_image = tf.expand_dims(input_image, 0)  # Add batch dimension
        input_image = input_image / 255.0  # Normalize to [0, 1] range

        # Make prediction
        predictions = model.predict(input_image)
        result = tf.nn.softmax(predictions[0])
        class_index = np.argmax(result)
        confidence = np.max(result) * 100
        return f"The image is classified as **{flower_names[class_index]}** with a confidence score of **{confidence:.2f}%**."
    except Exception as e:
        st.error(f"Error during classification: {e}")
        return None

# Image uploader for Streamlit
uploaded_file = st.file_uploader('Upload an Image of a Flower (JPEG/PNG)', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    try:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        # Read and preprocess the uploaded image
        file_bytes = uploaded_file.read()
        input_image = tf.image.decode_image(file_bytes, channels=3)

        # Ensure the image is valid
        if model:
            result = classify_image(input_image, model)
            if result:
                st.write(result)
    except Exception as e:
        st.error(f"Error processing the uploaded file: {e}")
