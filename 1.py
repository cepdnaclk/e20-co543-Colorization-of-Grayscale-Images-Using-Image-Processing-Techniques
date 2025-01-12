import cv2
import numpy as np

def colorize_image(image_path):
    # Load the grayscale image
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Check if the image was loaded properly
    if gray_image is None:
        print("Error: Could not load image.")
        return
    
    # Convert the grayscale image to a 3-channel image
    color_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    
    # Apply a colormap to the image
    color_image = cv2.applyColorMap(color_image, cv2.COLORMAP_JET)
    
    # Display the original and colorized images
    cv2.imshow('Grayscale Image', gray_image)
    cv2.imshow('Colorized Image', color_image)
    
    # Wait for a key press and close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
colorize_image('path_to_your_grayscale_image.jpg')