import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, BatchNormalization, Activation
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.callbacks import ModelCheckpoint

# Ensure TensorFlow uses GPU efficiently
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print("GPU is available and configured.")
else:
    print("No GPU found. Using CPU.")

# Load and preprocess images
def load_images(folder, img_size=(256, 256)):
    if not os.path.exists(folder):
        raise FileNotFoundError(f"The directory {folder} does not exist.")
    
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = load_img(img_path, target_size=img_size)
        img = img_to_array(img) / 255.0
        images.append(img)
    return np.array(images)

# Convert to LAB and separate channels
def preprocess_images(gray_folder, color_folder, img_size=(256, 256)):
    gray_images = load_images(gray_folder, img_size)
    color_images = load_images(color_folder, img_size)
    
    gray_images = np.expand_dims(gray_images[:, :, :, 0], axis=-1)
    
    lab_images = np.array([cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB) for img in color_images])
    ab_images = lab_images[:, :, :, 1:] / 128.0 - 1.0  # Normalize to [-1, 1]
    
    return gray_images, ab_images, lab_images

# Define CNN model for colorization
def build_model(input_shape):
    input_img = Input(shape=input_shape)
    
    # Encoder
    x = Conv2D(64, (3, 3), padding='same')(input_img)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same', strides=2)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same', strides=2)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Decoder
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(2, (3, 3), activation='tanh', padding='same')(x)
    
    return Model(input_img, x)

# Train model
def train_model(model, gray_images, color_images, epochs=100, batch_size=32):
    model.compile(optimizer='adam', loss='mse')
    checkpoint = ModelCheckpoint('colorization_model.h5', save_best_only=True, monitor='val_loss', mode='min')
    model.fit(gray_images, color_images, validation_split=0.1, epochs=epochs, batch_size=batch_size, callbacks=[checkpoint])

# Predict color channels
def colorize_images(model, gray_images):
    predicted_ab = model.predict(gray_images)
    return np.clip((predicted_ab + 1) * 128.0, 0, 255)  # Denormalize to [0, 255]

# Display results
def display_results(gray_images, predicted_ab, original_lab):
    for i in range(len(gray_images)):
        lab_image = np.zeros((256, 256, 3), dtype=np.uint8)
        lab_image[:, :, 0] = (gray_images[i].reshape(256, 256) * 255).astype(np.uint8)
        lab_image[:, :, 1:] = predicted_ab[i].astype(np.uint8)
        colorized_rgb = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)
        original_rgb = cv2.cvtColor(original_lab[i].astype(np.uint8), cv2.COLOR_LAB2RGB)
        
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(gray_images[i].reshape(256, 256), cmap='gray')
        plt.title('Grayscale')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(colorized_rgb)
        plt.title('Colorized')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(original_rgb)
        plt.title('Original')
        plt.axis('off')
        
        plt.show()

# Paths for dataset
gray_folder = "datasets/test/gray"
color_folder = "datasets/test/original"

# Preprocess images
gray_images, ab_images, lab_images = preprocess_images(gray_folder, color_folder)

# Build and train the model
input_shape = (256, 256, 1)
model = build_model(input_shape)
train_model(model, gray_images, ab_images)

# Load test images for colorization
test_gray_folder = "datasets/fruits/fruits_grayscale"
test_color_folder = "datasets/fruits/fruits_color"

test_gray_images, _, test_lab_images = preprocess_images(test_gray_folder, test_color_folder)

# Predict colorized images
predicted_ab = colorize_images(model, test_gray_images)

# Display results
display_results(test_gray_images, predicted_ab, test_lab_images)
