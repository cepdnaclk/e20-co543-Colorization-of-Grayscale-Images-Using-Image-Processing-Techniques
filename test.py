import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

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

# Adaptive contrast enhancement
def enhance_contrast(image, method="CLAHE"):
    if method == "CLAHE":
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))  # Adjusted clipLimit for better contrast
        return clahe.apply(image)
    elif method == "HistogramEq":
        return cv2.equalizeHist(image)
    else:
        return image  # Return original if no method is specified
# Example usage
dataset_path = "C:\Users\DELL\Documents\sem5\CO543\mini project\e20-co543-Colorization-of-Grayscale-Images-Using-Image-Processing-Techniques\datasets\test\original"
image_pairs = load_images(dataset_path)


# Edge detection with adaptive smoothing
def edge_detection(image):
    smoothed = cv2.bilateralFilter(image, 9, 70, 70)  # Smooth while preserving edges
    edges = cv2.Canny(smoothed, 100, 200)
    return edges

# Fourier Transform
def apply_fourier_transform(image):
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)
    return magnitude_spectrum

# Custom color mapping with blending
def apply_color_mapping(image, colormap=cv2.COLORMAP_TURBO, blend_ratio=0.7):
    enhanced_gray = enhance_contrast(image)  # Apply CLAHE
    colored = cv2.applyColorMap(enhanced_gray, colormap)
    
    # Blend grayscale with colorized output for a more natural effect
    blended = cv2.addWeighted(colored, blend_ratio, cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), 1 - blend_ratio, 0)
    
    return blended

# Load dataset
dataset_path = "datasets/Flowers_Test/Original"
image_pairs = load_images(dataset_path)

# Ensure we have images
if not image_pairs:
    raise ValueError("No images found in the dataset folder!")

# Select the first grayscale image and original image
gray_image = image_pairs[0][0]
original_image = image_pairs[0][1]

# Apply edge detection
edges = edge_detection(gray_image)

# Apply Fourier Transform
fourier_image = apply_fourier_transform(gray_image)

# Apply different colormaps
colormaps = [
    (cv2.COLORMAP_TURBO, "TURBO"),
    (cv2.COLORMAP_INFERNO, "INFERNO"),
    (cv2.COLORMAP_JET, "JET"),
    (cv2.COLORMAP_HOT, "HOT"),
    (cv2.COLORMAP_VIRIDIS, "VIRIDIS"),
    (cv2.COLORMAP_RAINBOW, "RAINBOW")
]

# Create a figure to display all images
plt.figure(figsize=(18, 12))  # Increased figure size to accommodate all images

# Plot original image
plt.subplot(3, 4, 1)
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')

# Plot grayscale image
plt.subplot(3, 4, 2)
plt.imshow(gray_image, cmap='gray')
plt.title("Grayscale Image")
plt.axis('off')

# Plot edges
plt.subplot(3, 4, 3)
plt.imshow(edges, cmap='gray')
plt.title("Edges with Adaptive Smoothing")
plt.axis('off')

# Plot Fourier Transform
plt.subplot(3, 4, 4)
plt.imshow(fourier_image, cmap='gray')
plt.title("Fourier Transform")
plt.axis('off')

# Plot enhanced color-mapped images
for i, (cmap, cmap_name) in enumerate(colormaps):
    colored_image = apply_color_mapping(gray_image, cmap, blend_ratio=0.7)
    plt.subplot(3, 4, 5 + i)  # Adjusted the starting index for the colormaps
    plt.imshow(cv2.cvtColor(colored_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Enhanced - {cmap_name}")
    plt.axis('off')

# Adjust layout and display
plt.tight_layout()
plt.show()
