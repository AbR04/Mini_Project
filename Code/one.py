import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# Load the pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Function to load and preprocess the image
def load_and_preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, (224, 224))  # Resize to the input size of the model
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = preprocess_input(img)  # Preprocess for MobileNetV2
    return img

# Function to predict disease
def predict_disease(img_path):
    img = load_and_preprocess_image(img_path)
    preds = model.predict(img)
    decoded_preds = decode_predictions(preds, top=3)[0]  # Get top 3 predictions
    return decoded_preds

# Load and predict disease from an X-ray image
image_path = 'xray_1.jpg'  # Update this path
predictions = predict_disease(image_path)

# Display the results
print("Predictions:")
for pred in predictions:
    print(f"{pred[1]}: {pred[2]*100:.2f}%")

# Display the image
xray_image = cv2.imread(image_path)
xray_image = cv2.cvtColor(xray_image, cv2.COLOR_BGR2RGB)  # Convert to RGB
plt.imshow(xray_image)
plt.title('X-ray Image')
plt.axis('off')
plt.show()
