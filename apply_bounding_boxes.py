import cv2
import json
import os

# Define the dataset paths
dataset_path = r"C:\Users\TECQNIO\project_folder\dataset"
label_file = os.path.join(dataset_path, "labels.json")  # Path to your saved labels

# Function to draw bounding boxes on images
def draw_bounding_boxes():
    # Load labels from the JSON file
    try:
        with open(label_file, 'r') as file:
            labels = json.load(file)
    except FileNotFoundError:
        print(f"Error: Could not find {label_file}. Make sure it exists.")
        return

    # Loop through each labeled image
    for image_name, boxes in labels.items():
        # Determine if the image is in the 'good' or 'bad' folder
        if image_name.startswith("good"):
            image_path = os.path.join(dataset_path, "good", image_name)
        elif image_name.startswith("bad"):
            image_path = os.path.join(dataset_path, "bad", image_name)
        else:
            print(f"Error: Could not categorize {image_name}.")
            continue

        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image at {image_path}. Skipping.")
            continue

        # Draw each bounding box on the image
        for box in boxes:
            x, y, w, h = box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the image with bounding boxes
        cv2.imshow("Labeled Image", image)
        cv2.waitKey(0)  # Press any key to close the image window

    cv2.destroyAllWindows()

# Run the function
draw_bounding_boxes()
