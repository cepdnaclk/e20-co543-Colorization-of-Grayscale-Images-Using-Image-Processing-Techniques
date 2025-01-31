import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Load and preprocess images
def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                images.append((gray, img))  # Store both grayscale and original
            else:
                print(f"Warning: Unable to read {img_path}")
        else:
            print(f"Warning: File not found - {img_path}")
    return images

# Create a custom colormap
def create_custom_colormap():
    # Define a list of colors in RGB format
    colors = [(1.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 1.0), (0.0, 0.0, 0.0)]  # Black -> Blue -> Red -> Yellow -> Green

    # Create a colormap with these colors
    n_bins = 20  # Number of bins to divide the color spectrum into
    cmap_name = 'custom_cmap'
    cm = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    
    return cm

# Load dataset
dataset_path = "datasets/Flowers_Test/Original"
image_pairs = load_images(dataset_path)

# Ensure we have images
if not image_pairs:
    raise ValueError("No images found in the dataset folder!")

# Create custom colormap
custom_cmap = create_custom_colormap()

# Create a figure to display all images with increased figure size
fig_width = 100
fig_height = len(image_pairs) * 3  # Adjust the height to accommodate more rows

plt.figure(figsize=(fig_width, fig_height))

# Iterate through all the images in the dataset
for i, (gray_image, original_image) in enumerate(image_pairs):
    # Normalize the grayscale image for colormap
    norm = mcolors.Normalize(vmin=np.min(gray_image), vmax=np.max(gray_image))

    # Apply custom colormap using matplotlib's colormap and normalization
    colored_image_custom = custom_cmap(norm(gray_image))  # Applying custom colormap

    # Convert the colored image to a format OpenCV can handle (BGR)
    colored_image_custom_bgr = (colored_image_custom[:, :, :3] * 255).astype(np.uint8)

    # Plot original image
    plt.subplot(len(image_pairs), 3, i * 3 + 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Original Image {i+1}")
    plt.axis('off')

    # Plot grayscale image
    plt.subplot(len(image_pairs), 3, i * 3 + 2)
    plt.imshow(gray_image, cmap='gray')
    plt.title(f"Grayscale Image {i+1}")
    plt.axis('off')

    # Plot colorized image with custom colormap
    plt.subplot(len(image_pairs), 3, i * 3 + 3)
    plt.imshow(cv2.cvtColor(colored_image_custom_bgr, cv2.COLOR_BGR2RGB))
    plt.title(f"Custom Color Map Colorized {i+1}")
    plt.axis('off')

# Adjust layout and display
plt.tight_layout()
plt.show()
