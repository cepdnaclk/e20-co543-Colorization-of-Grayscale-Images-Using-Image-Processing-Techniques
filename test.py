import cv2
import numpy as np
import os

# Load and preprocess images
def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            images.append((gray, img))  # Store both grayscale and original
    return images

# Example usage
dataset_path = "C:\Users\DELL\Documents\sem5\CO543\mini project\e20-co543-Colorization-of-Grayscale-Images-Using-Image-Processing-Techniques\datasets\test\original"
image_pairs = load_images(dataset_path)


def edge_detection(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)
    return edges

# Example usage
gray_image = image_pairs[0][0]  # First grayscale image
edges = edge_detection(gray_image)

cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

def apply_fourier_transform(image):
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)
    return magnitude_spectrum

# Example usage
fourier_image = apply_fourier_transform(gray_image)

cv2.imshow('Fourier Transform', fourier_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

def apply_color_mapping(image):
    colored = cv2.applyColorMap(image, cv2.COLORMAP_JET)  # Example colormap
    return colored

# Example usage
colored_image = apply_color_mapping(gray_image)

cv2.imshow('Colorized Image', colored_image)
cv2.waitKey(0)
cv2.destroyAllWindows()





