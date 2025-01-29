import cv2
import numpy as np
import os

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

# Display results side by side
def display_results(filename, original, gray, edges, fourier, colorized):
    # Resize images to a standard size for consistent visualization
    standard_size = (256, 256)
    original_resized = cv2.resize(original, standard_size)
    gray_resized = cv2.resize(gray, standard_size)
    edges_resized = cv2.resize(edges, standard_size)
    fourier_resized = cv2.resize(fourier, standard_size)
    colorized_resized = cv2.resize(colorized, standard_size)

    # Convert grayscale images to BGR for visualization
    gray_resized = cv2.cvtColor(gray_resized, cv2.COLOR_GRAY2BGR)
    edges_resized = cv2.cvtColor(edges_resized, cv2.COLOR_GRAY2BGR)
    fourier_resized = cv2.cvtColor(fourier_resized.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Stack images horizontally
    combined = np.hstack((original_resized, gray_resized, edges_resized, fourier_resized, colorized_resized))
    
    # Display the images
    cv2.imshow(f"Results for {filename}", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Define dataset path
dataset_path = "D:/Academic/CO543 - Image Processing/e20-co543-Colorization-of-Grayscale-Images-Using-Image-Processing-Techniques/datasets/test/original"

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
        
        display_results(filename, original_image, gray_image, edges, fourier_image, colorized_image)
