import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Load and preprocess images
def load_images(gray_folder, color_folder):
    images = []
    gray_filenames = os.listdir(gray_folder)
    color_filenames = os.listdir(color_folder)
    
    for gray_filename, color_filename in zip(gray_filenames, color_filenames):
        gray_img_path = os.path.join(gray_folder, gray_filename)
        color_img_path = os.path.join(color_folder, color_filename)
        
        print(f"Trying to load: {gray_img_path} and {color_img_path}")  # Debugging statement
        
        gray_img = cv2.imread(gray_img_path, cv2.IMREAD_GRAYSCALE)
        color_img = cv2.imread(color_img_path)
        
        if gray_img is not None and color_img is not None:
            images.append((gray_img, color_img))  # Store both grayscale and original
        else:
            print(f"Warning: Could not load images {gray_img_path} or {color_img_path}")  # Debugging statement
    
    return images

# Apply color mapping
def apply_color_mapping(image):
    colored = cv2.applyColorMap(image, cv2.COLORMAP_JET)  # Example colormap
    return colored

# Example usage
gray_folder = "datasets/fruits/fruits_grayscale"
color_folder = "datasets/fruits/fruits_color"

image_pairs = load_images(gray_folder, color_folder)

# Plotting the images in pairs
for gray_img, color_img in image_pairs:
    colored_image = apply_color_mapping(gray_img)
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(gray_img, cmap='gray')
    ax[0].set_title('Grayscale Image')
    ax[1].imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
    ax[1].set_title('Original Image')
    ax[2].imshow(cv2.cvtColor(colored_image, cv2.COLOR_BGR2RGB))
    ax[2].set_title('Colorized Image')
    
    # Hide axes
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    
    plt.show()