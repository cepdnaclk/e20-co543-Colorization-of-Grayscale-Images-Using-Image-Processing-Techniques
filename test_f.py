import cv2
import numpy as np
import os

# Load reference color image
def load_reference_image(reference_path):
    ref_img = cv2.imread(reference_path)
    if ref_img is None:
        print("Error: Reference image not found!")
        return None
    return cv2.cvtColor(ref_img, cv2.COLOR_BGR2LAB)  # Convert to LAB color space

# Histogram matching for color transfer
def color_transfer(source_gray, reference_color):
    source_lab = cv2.cvtColor(cv2.cvtColor(source_gray, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2LAB)

    for i in range(1, 3):  # Only apply to A and B channels (not L)
        source_hist, _ = np.histogram(source_lab[:, :, i], bins=256, range=(0, 256), density=True)
        reference_hist, _ = np.histogram(reference_color[:, :, i], bins=256, range=(0, 256), density=True)

        # Compute cumulative distribution function (CDF)
        source_cdf = np.cumsum(source_hist)
        reference_cdf = np.cumsum(reference_hist)

        # Use CDF mapping to transform source image colors
        lookup_table = np.interp(source_cdf, reference_cdf, np.arange(256))
        source_lab[:, :, i] = cv2.LUT(source_lab[:, :, i], lookup_table.astype('uint8'))

    return cv2.cvtColor(source_lab, cv2.COLOR_LAB2BGR)

# Texture-based segmentation (for better color application)
def segment_image(image):
    edges = cv2.Canny(image, 100, 200)
    mask = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)  # Expand edges
    return cv2.bitwise_not(mask)  # Invert mask to highlight regions

# Apply color mapping based on segmentation
def apply_custom_coloring(image, segmented_mask):
    colorized = cv2.applyColorMap(image, cv2.COLORMAP_TURBO)  # Base colormap
    colorized[segmented_mask == 255] = [0, 255, 0]  # Force green for segmented areas
    return colorized

# Load images
gray_image_path = "datasets/Flowers/flowers_grey/0001.png"
reference_image_path = "datasets/Flowers/flowers_colour/0001.png"

gray_image = cv2.imread(gray_image_path, cv2.IMREAD_GRAYSCALE)
reference_image = load_reference_image(reference_image_path)

if gray_image is None or reference_image is None:
    raise ValueError("Error: Could not load images!")

# Step 1: Apply color transfer
colorized_image = color_transfer(gray_image, reference_image)

# Step 2: Apply segmentation-based enhancement
segmented_mask = segment_image(gray_image)
final_output = apply_custom_coloring(colorized_image, segmented_mask)

# Display results
cv2.imshow("Grayscale Image", gray_image)
cv2.imshow("Colorized Image (Classical Approach)", final_output)
cv2.waitKey(0)
cv2.destroyAllWindows()
