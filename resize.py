import cv2
import os

# Define paths to the good and bad image folders
good_folder = r"C:\Users\TECQNIO\project_folder\dataset\good"
bad_folder = r"C:\Users\TECQNIO\project_folder\dataset\bad"

# Define desired width and height
new_width = 640
new_height = 400

def resize_images(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Load image
        image = cv2.imread(file_path)
        if image is None:
            print(f"Error: Could not load image at {file_path}. Skipping.")
            continue
        
        # Resize image
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Save resized image, overwriting the original
        cv2.imwrite(file_path, resized_image)
        print(f"Resized and saved {filename}")

# Resize images in both folders
resize_images(good_folder)
resize_images(bad_folder)

print("All images resized successfully.")
