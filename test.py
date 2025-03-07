import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Create multiple custom colormaps
def create_custom_colormap():
    colors1 = [(1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]  # Red to Yellow to Green
    colors2 = [(0.0, 0.0, 1.0), (0.0, 1.0, 1.0), (1.0, 0.0, 1.0)]  # Blue to Cyan to Magenta
    colors3 = [(0.5, 0.0, 0.5), (1.0, 0.5, 0.0), (0.0, 0.5, 1.0)]  # Purple to Orange to Blue
    colors4 = [(0.0, 1.0, 0.5), (0.5, 0.0, 1.0), (1.0, 1.0, 0.5)]  # Greenish to Violet to Yellowish
    colors5 = [(1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0)]  # Light Red to Light Green to Light Blue
    
    cmap1 = mcolors.LinearSegmentedColormap.from_list("custom_cmap1", colors1, N=256)
    cmap2 = mcolors.LinearSegmentedColormap.from_list("custom_cmap2", colors2, N=256)
    cmap3 = mcolors.LinearSegmentedColormap.from_list("custom_cmap3", colors3, N=256)
    cmap4 = mcolors.LinearSegmentedColormap.from_list("custom_cmap4", colors4, N=256)
    cmap5 = mcolors.LinearSegmentedColormap.from_list("custom_cmap5", colors5, N=256)
    return cmap1, cmap2, cmap3, cmap4, cmap5

# Map intensity to color manually
def map_intensity_to_color(intensity, scheme):
    if scheme == 1:
        return [255, 0, 0] if intensity < 85 else [0, 255, 0] if intensity < 170 else [0, 0, 255]
    elif scheme == 2:
        return [intensity, 255 - intensity, 128]
    elif scheme == 3:
        return [255 - intensity, intensity // 2, intensity]
    elif scheme == 4:
        return [intensity // 2, intensity, 255 - intensity]
    elif scheme == 5:
        return [255 - intensity, 128, intensity // 2]
    else:
        return [intensity, intensity, intensity]

# Apply intensity mapping
def apply_manual_mapping(image, scheme):
    height, width = image.shape
    color_mapped = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            color_mapped[i, j] = map_intensity_to_color(image[i, j], scheme)
    return color_mapped

# Apply OpenCV colormaps
def apply_opencv_colormap(image, colormap):
    return cv2.applyColorMap(image, colormap)

# Main function to process and display images
def process_and_display(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale

    # Apply mappings
    cmap1, cmap2, cmap3, cmap4, cmap5 = create_custom_colormap()
    custom_mapped1 = (cmap1(image / 255.0)[:, :, :3] * 255).astype(np.uint8)
    custom_mapped2 = (cmap2(image / 255.0)[:, :, :3] * 255).astype(np.uint8)
    custom_mapped3 = (cmap3(image / 255.0)[:, :, :3] * 255).astype(np.uint8)
    custom_mapped4 = (cmap4(image / 255.0)[:, :, :3] * 255).astype(np.uint8)
    custom_mapped5 = (cmap5(image / 255.0)[:, :, :3] * 255).astype(np.uint8)
    
    manual_mapped1 = apply_manual_mapping(image, 1)
    manual_mapped2 = apply_manual_mapping(image, 2)
    manual_mapped3 = apply_manual_mapping(image, 3)
    manual_mapped4 = apply_manual_mapping(image, 4)
    manual_mapped5 = apply_manual_mapping(image, 5)

    opencv_mapped1 = apply_opencv_colormap(image, cv2.COLORMAP_JET)
    opencv_mapped2 = apply_opencv_colormap(image, cv2.COLORMAP_HOT)
    opencv_mapped3 = apply_opencv_colormap(image, cv2.COLORMAP_PARULA)
    opencv_mapped4 = apply_opencv_colormap(image, cv2.COLORMAP_VIRIDIS)
    opencv_mapped5 = apply_opencv_colormap(image, cv2.COLORMAP_MAGMA)

    # Display images
    plt.figure(figsize=(18, 12))
    titles = [
         "Custom Colormap 1", "Custom Colormap 2", "Custom Colormap 3", "Custom Colormap 4", "Custom Colormap 5",
        "Manual Mapping 1", "Manual Mapping 2", "Manual Mapping 3", "Manual Mapping 4", "Manual Mapping 5",
        "OpenCV Jet", "OpenCV Hot", "OpenCV Parula", "OpenCV Viridis", "OpenCV Magma"
    ]
    images = [
         custom_mapped1, custom_mapped2, custom_mapped3, custom_mapped4, custom_mapped5,
        manual_mapped1, manual_mapped2, manual_mapped3, manual_mapped4, manual_mapped5,
        opencv_mapped1, opencv_mapped2, opencv_mapped3, opencv_mapped4, opencv_mapped5
    ]

    for i in range(15):
        plt.subplot(3,5,i + 1)
        
        plt.imshow(images[i])
        plt.title(titles[i], fontsize=8)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# Example usage
process_and_display("D:/Academic/CO543 - Image Processing/e20-co543-Colorization-of-Grayscale-Images-Using-Image-Processing-Techniques/datasets/Flowers/flowers_colour/0008.png")
