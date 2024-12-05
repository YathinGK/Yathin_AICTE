import os
import tensorflow as tf
import numpy as np
from keras.models import load_model
import streamlit as st

# Streamlit App Header
st.header('Flower Classification')

# Flower categories
flower_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Upload the model dynamically
model_file = st.file_uploader('Upload the Trained Keras Model (.keras)', type=['keras'])

if model_file is not None:
    # Save the uploaded model temporarily
    with open("uploaded_model.keras", "wb") as f:
        f.write(model_file.getbuffer())
    try:
        model = load_model("uploaded_model.keras")
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    # Classification function
    def classify_images(image):
        # Resize and preprocess image
        input_image = tf.image.resize(image, (180, 180))
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
        try:
            file_bytes = uploaded_file.read()
            input_image = tf.image.decode_image(file_bytes, channels=3)

            # Display classification result
            result = classify_images(input_image)
            st.write(result)
        except Exception as e:
            st.error(f"Error processing image: {e}")
else:
    st.warning("Please upload a trained Keras model file to proceed.")
