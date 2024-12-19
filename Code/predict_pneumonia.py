import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Load the trained model
model = load_model('xray_model.h5')

# Function to load and preprocess the image
def load_and_preprocess_image(img_path):
    # Check if the file exists
    if not os.path.exists(img_path):
        print(f"Error: The file '{img_path}' does not exist.")
        return None
    
    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))  # Resize image to match input size of model
    img_array = tf.keras.preprocessing.image.img_to_array(img)                        # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)                                   # Add batch dimension (model expects 4D input)
    img_array = img_array / 255.0                                                   # Normalize the image (model trained on [0, 1] range)
    return img_array

# Path to the new image
new_image_path = 'NEW/5.jpg'  # Update with the correct path to your test image
preprocessed_image = load_and_preprocess_image(new_image_path)

# Proceed only if the image was successfully loaded
if preprocessed_image is not None:
    # Make predictions
    predictions = model.predict(preprocessed_image)

    # Interpret the predictions (for binary classification)
    predicted_class = (predictions[0][0] > 0.5).astype("int32")  # Thresholding at 0.5

    # Class labels dictionary
    class_labels = {0: 'Normal', 1: 'Pneumonia'}

    # Print the result
    print(f'The model predicts: {class_labels[predicted_class]}')
