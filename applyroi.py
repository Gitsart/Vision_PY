import cv2
import os
import json
import numpy as np

# Load the saved ROIs
with open("rois.json", "r") as file:
    rois = json.load(file)

def preprocess_image_with_rois(image_path, rois):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Crop regions using ROIs
    cropped_regions = []
    for x, y, w, h in rois:
        roi = gray[y:y+h, x:x+w]  # Crop each ROI
        cropped_regions.append(cv2.resize(roi, (32, 32)))  # Resize to 32x32 pixels

    # Combine all ROIs into a single image (optional)
    processed_image = np.hstack(cropped_regions)
    
    return processed_image

def process_dataset_with_rois(input_folder, output_folder, rois):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        image_path = os.path.join(input_folder, filename)
        processed_image = preprocess_image_with_rois(image_path, rois)
        
        if processed_image is not None:
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, processed_image)
            print(f"Processed and saved: {output_path}")

# Process 'good' and 'bad' datasets using ROIs
process_dataset_with_rois(r"C:/Users/TECQNIO/project_folder/dataset/good", "processed_dataset/good", rois)
process_dataset_with_rois(r"C:/Users/TECQNIO/project_folder/dataset/bad", "processed_dataset/bad", rois)
