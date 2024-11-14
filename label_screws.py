import cv2
import json
import os

# Set paths
dataset_path = "C:/Users/TECQNIO/project_folder/dataset"
label_file_path = os.path.join(dataset_path, "labels.json")

# Load or create labels JSON file
if os.path.exists(label_file_path):
    with open(label_file_path, "r") as file:
        labels = json.load(file)
else:
    labels = {}

# Initialize global variables
drawing = False
ix, iy = -1, -1
bbox_list = []

# Mouse callback function
def draw_box(event, x, y, flags, param):
    global ix, iy, drawing, bbox_list, image, clone

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            image = clone.copy()  # Reset to the original image to avoid overlapping rectangles
            cv2.rectangle(image, (ix, iy), (x, y), (0, 255, 0), 2)
            for (x1, y1, w, h) in bbox_list:  # Draw previously saved boxes
                cv2.rectangle(image, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        width, height = x - ix, y - iy
        bbox_list.append((ix, iy, width, height))  # Save (x, y, w, h) for the box
        cv2.rectangle(image, (ix, iy), (x, y), (0, 255, 0), 2)

# Process images in both 'good' and 'bad' folders
for category in ["good", "bad"]:
    category_path = os.path.join(dataset_path, category)
    for filename in os.listdir(category_path):
        if filename in labels:
            continue  # Skip if already labeled

        filepath = os.path.join(category_path, filename)
        image = cv2.imread(filepath)
        if image is None:
            print(f"Could not read {filepath}. Skipping.")
            continue

        clone = image.copy()
        bbox_list = []  # Reset bounding boxes for each new image

        # Set up window and callback
        cv2.namedWindow("Label Screws")
        cv2.setMouseCallback("Label Screws", draw_box)

        # Show image and mark bounding boxes
        while True:
            cv2.imshow("Label Screws", image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):  # Save labels and move to the next image
                if bbox_list:
                    labels[filename] = bbox_list
                    with open(label_file_path, "w") as file:
                        json.dump(labels, file, indent=4)
                    print(f"Saved bounding boxes for {filename}.")
                break

            elif key == ord("r"):  # Reset current bounding boxes for the image
                image = clone.copy()
                bbox_list = []
                print("Reset bounding boxes for this image.")

            elif key == 27:  # Press 'Esc' to skip this image
                print(f"Skipping {filename}.")
                break

        cv2.destroyAllWindows()
