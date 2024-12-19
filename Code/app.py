import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load your trained model
MODEL_PATH = 'xray_model.h5'
model = load_model(MODEL_PATH)

# Define class labels
CLASS_LABELS = {0: 'Normal', 1: 'Pneumonia'}

# Function to preprocess the image
def preprocess_image(img_path):
    img = load_img(img_path, target_size=(224, 224))  # Resize image
    img_array = img_to_array(img)                     # Convert to array
    img_array = np.expand_dims(img_array, axis=0)     # Add batch dimension
    img_array = img_array / 255.0                     # Normalize
    return img_array

# Streamlit UI
st.title("Chest X-ray Classification")
st.write("Upload a chest X-ray image to classify it as Normal or Pneumonia.")

# File uploader
uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded X-ray Image", use_column_width=True)
    st.write("Processing...")

    # Save the uploaded file to a temporary location
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Preprocess the image
    preprocessed_image = preprocess_image("temp_image.jpg")

    # Make prediction
    predictions = model.predict(preprocessed_image)
    predicted_class = (predictions[0][0] > 0.5).astype("int32")
    confidence = predictions[0][0] if predicted_class == 1 else 1 - predictions[0][0]

    # Display the result
    prediction_label = CLASS_LABELS[predicted_class]
    st.write(f"**Prediction:** {prediction_label}")
    st.write(f"**Confidence:** {confidence * 100:.2f}%")

    # Display a health message based on the prediction
    if predicted_class == 1:  # Pneumonia
        st.error("Visit a doctor soon for further evaluation.")
    else:  # Normal
        st.success("You are healthy. Keep up the good work!")
