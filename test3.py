import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Load and preprocess images
def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):  # Ensure it's a valid file
            img = cv2.imread(img_path)
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                images.append((filename, gray, img))  # Store filename, grayscale, and original
    return images

# Apply Edge Detection
def edge_detection(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)
    return edges

# Apply Fourier Transform
def apply_fourier_transform(image):
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)
    return magnitude_spectrum

# Apply Color Mapping
def apply_color_mapping(image):
    colored = cv2.applyColorMap(image, cv2.COLORMAP_JET)  # Example colormap
    return colored

# Display results in a single figure
def plot_results(filename, original, gray, edges, fourier, colorized):
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))  # 1 row, 5 columns
    
    # Convert BGR to RGB for Matplotlib
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    colorized_rgb = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)
    
    # Titles and images
    titles = ['Original Image', 'Grayscale Image', 'Edge Detection', 'Fourier Transform', 'Colorized Image']
    images = [original_rgb, gray, edges, fourier, colorized_rgb]
    
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)  # Use grayscale for single-channel images
        ax.set_title(title)
        ax.axis('off')  # Hide axes
    
    # Set main title
    plt.suptitle(f"Processing Results for {filename}", fontsize=14)
    plt.tight_layout()
    plt.show()

# Define dataset path
dataset_path = r"C:\Users\DELL\Documents\sem5\CO543\mini project\e20-co543-Colorization-of-Grayscale-Images-Using-Image-Processing-Techniques\datasets\test\cartoon"
#dataset_path = "C:\Users\DELL\Documents\sem5\CO543\mini project\e20-co543-Colorization-of-Grayscale-Images-Using-Image-Processing-Techniques\datasets\test\original"

# Load images
image_pairs = load_images(dataset_path)

# Process each image
if not image_pairs:
    print("Error: No valid images found in the dataset folder.")
else:
    for filename, gray_image, original_image in image_pairs:
        edges = edge_detection(gray_image)
        fourier_image = apply_fourier_transform(gray_image)
        colorized_image = apply_color_mapping(gray_image)
        
        plot_results(filename, original_image, gray_image, edges, fourier_image, colorized_image)
