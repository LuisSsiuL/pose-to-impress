import cv2
import numpy as np
import mediapipe as mp
import os

# Initialize Mediapipe Selfie Segmentation for background removal
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Paths to the input and output folders
input_folder = './pose_images'
output_folder = './outlined_images'

# Check if output folder exists, create if not
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to remove background and create an outline using edge detection
def create_outline(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert to RGB for Mediapipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform segmentation to remove background
    results = segmentation.process(image_rgb)
    mask = results.segmentation_mask > 0.5  # Mask for foreground

    # Create a transparent background image (4 channels: RGBA)
    transparent_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)

    # Extract the foreground (the person) using the segmentation mask
    foreground = np.zeros_like(image, dtype=np.uint8)
    foreground[mask] = image[mask]

    # Convert foreground to grayscale for edge detection
    gray_foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to smooth the image for better edge detection
    blurred = cv2.GaussianBlur(gray_foreground, (5, 5), 0)

    # Perform Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Create a mask for edges
    edge_mask = edges > 0

    # Set edges to white in the transparent image with full opacity (255)
    transparent_image[edge_mask] = [255, 255, 255, 255]

    # Save the resulting image with transparent background
    cv2.imwrite(output_path, transparent_image)

# Process all images in the folder
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, 'outlined_' + filename)
        create_outline(input_path, output_path)

print(f"Outlined images saved in {output_folder}")
