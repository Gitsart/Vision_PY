import cv2
import pandas as pd
import os

# Define a function to save the annotations
def save_annotations(image_path, bboxes, labels, save_path):
    annotations = []
    for i, bbox in enumerate(bboxes):
        # xmin, ymin, xmax, ymax
        annotations.append({
            'image': image_path,
            'xmin': bbox[0],
            'ymin': bbox[1],
            'xmax': bbox[2],
            'ymax': bbox[3],
            'label': labels[i]
        })

    # Create a DataFrame to save to CSV
    df = pd.DataFrame(annotations)
    df.to_csv(save_path, mode='a', header=not os.path.exists(save_path), index=False)

# Create a function to handle the manual labeling process
def label_image(image_path, label_name):
    image = cv2.imread(image_path)
    clone = image.copy()
    bboxes = []
    labels = []
    is_drawing = False
    ix, iy = -1, -1

    def draw_rectangle(event, x, y, flags, param):
        nonlocal ix, iy, is_drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            is_drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if is_drawing:
                image_copy = clone.copy()
                cv2.rectangle(image_copy, (ix, iy), (x, y), (0, 255, 0), 2)
                cv2.imshow("Labeling", image_copy)
        elif event == cv2.EVENT_LBUTTONUP:
            is_drawing = False
            bboxes.append((ix, iy, x, y))  # Save the bounding box coordinates
            labels.append(label_name)  # Label it as "Screw"
            cv2.rectangle(image, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow("Labeling", image)

    cv2.imshow("Labeling", image)
    cv2.setMouseCallback("Labeling", draw_rectangle)

    print("Press 'r' to reset the box, 'q' to quit labeling.")
    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):  # Quit the labeling process
            break
        elif k == ord('r'):  # Reset bounding boxes
            image = clone.copy()
            bboxes = []
            labels = []

    # Save the annotations to CSV after labeling is done
    save_annotations(image_path, bboxes, labels, 'annotations.csv')
    cv2.destroyAllWindows()

# Loop through all images in the 'images/' directory and label them
image_dir = 'images/'  # Path to your image directory
for image_name in os.listdir(image_dir):
    if image_name.endswith('.jpg') or image_name.endswith('.png'):
        image_path = os.path.join(image_dir, image_name)
        print(f"Labeling {image_path}...")
        label_image(image_path, label_name="Screw")

