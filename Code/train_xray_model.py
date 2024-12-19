# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import tensorflow as tf
# try:
#     layers = tf.keras.layers
#     models = tf.keras.models
#     ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
#     print("Keras imports successful!")
# except ImportError as e:
#     print(f"Import error: {e}")

# # Set paths for your dataset
# train_dir = 'XrayDataset/train'  # Update this path if needed
# validation_dir = 'XrayDataset/validation'  # Update this path if needed

# # Data augmentation and normalization
# train_datagen = ImageDataGenerator(
#     rescale=1.0/255,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )

# validation_datagen = ImageDataGenerator(rescale=1.0/255)

# # Load images from directories
# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='binary'  # Change to 'categorical' for multi-class
# )

# validation_generator = validation_datagen.flow_from_directory(
#     validation_dir,
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='binary'  # Change to 'categorical' for multi-class
# )

# # Load the pre-trained MobileNetV2 model
# base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
# base_model.trainable = False  # Freeze the base model

# # Add custom layers
# model = models.Sequential([
#     base_model,
#     layers.GlobalAveragePooling2D(),
#     layers.Dense(128, activation='relu'),
#     layers.Dense(1, activation='sigmoid')  # Change units for multi-class
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Adjust loss for multi-class

# # Fine-tune the model
# history = model.fit(
#     train_generator,
#     steps_per_epoch=train_generator.samples // train_generator.batch_size,
#     validation_data=validation_generator,
#     validation_steps=validation_generator.samples // validation_generator.batch_size,
#     epochs=10
# )

# # Save the model
# model.save('xray_model.h5')

# # Print training and validation metrics
# print("Training complete!")
# print(f"Final Training Accuracy: {history.history['accuracy'][-1]}")
# print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]}")

# # Plot training & validation accuracy values
# plt.plot(history.history['accuracy'], label='Train Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.title('Model Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(loc='upper left')
# plt.show()

# # Plot training & validation loss values
# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Model Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(loc='upper left')
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Check for Keras and TensorFlow imports
try:
    layers = tf.keras.layers
    models = tf.keras.models
    ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
    print("Keras imports successful!")
except ImportError as e:
    print(f"Import error: {e}")

# Paths for your datasets
train_dir = 'XrayDataset/train'  # Update path if needed
validation_dir = 'XrayDataset/validation'  # Update path if needed
test_dir = 'XrayDataset/test'  # Add path to your test dataset

# Data augmentation and normalization for training and validation
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1.0/255)
test_datagen = ImageDataGenerator(rescale=1.0/255)

# Load images from directories
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'  # Change to 'categorical' for multi-class
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'  # Change to 'categorical' for multi-class
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',  # Change to 'categorical' for multi-class
    shuffle=False  # Don't shuffle test data for evaluation
)

# Load the pre-trained MobileNetV2 model
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze the base model

# Add custom layers
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Change units for multi-class
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10
)

# Save the trained model
model.save('xray_model.h5')

# Print final training and validation accuracy
print("Training complete!")
print(f"Final Training Accuracy: {history.history['accuracy'][-1]}")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]}")

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

# Test the model (evaluate on the test set)
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Optionally, you can make predictions on individual images
# Example: Predict a single image
test_image_path = 'path_to_test_image.jpg'  # Update with a valid test image path
img = image.load_img(test_image_path, target_size=(224, 224))
img_array = image.img_to_array(img)  # Convert image to array
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Normalize the image
img_array = img_array / 255.0

# Make prediction
prediction = model.predict(img_array)

# Output the prediction
if prediction[0] > 0.5:
    print("Predicted Class: 1 (e.g., Disease present)")
else:
    print("Predicted Class: 0 (e.g., Healthy)")
