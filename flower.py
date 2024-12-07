import tensorflow as tf
import numpy as np
from keras.models import load_model
import streamlit as st
from PIL import Image
import io

# Streamlit App Title
st.title('🌼 Flower Classification Using CNN')

# Flower categories
flower_names = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']

# Load the trained model
@st.cache_resource  # Cache the model to improve performance in Streamlit
def load_flower_model(model_path):
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model_path = "Flower_Recog_Model.keras"
model = load_flower_model(model_path)

# Classification function
def classify_images(image):
    try:
        # Preprocess the image
        input_image = tf.image.resize(image, (180, 180))  # Resize to model's input size
        input_image = tf.expand_dims(input_image, 0)  # Add batch dimension
        input_image = tf.cast(input_image, tf.float32) / 255.0  # Normalize pixel values

        # Predict the class
        predictions = model.predict(input_image, verbose=0)
        result = tf.nn.softmax(predictions[0])  # Apply softmax for probabilities
        class_index = np.argmax(result)
        confidence = np.max(result) * 100

        # Return the result
        return f"The image is classified as **{flower_names[class_index]}** with a confidence score of **{confidence:.2f}%**."
    except Exception as e:
        return f"Error during classification: {e}"

# File uploader for image
uploaded_file = st.file_uploader("Upload an Image of a Flower (JPEG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display the uploaded image
        image_bytes = uploaded_file.read()
        input_image = tf.image.decode_image(image_bytes, channels=3)
        st.image(Image.open(io.BytesIO(image_bytes)), caption="Uploaded Image", use_column_width=True)

        # Classify the image
        with st.spinner("Classifying..."):
            result = classify_images(input_image)

        # Display the result
        st.success(result)
    except Exception as e:
        st.error(f"Error processing the uploaded image: {e}")
else:
    st.info("Please upload an image to classify.")
