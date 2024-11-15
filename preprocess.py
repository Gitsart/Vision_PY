import os
import cv2
import json
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle

# Step 1: Load Images and Labels
good_dir = "processed_dataset/good"
bad_dir = "processed_dataset/bad"

# Load ROIs for reference
with open("rois.json", "r") as file:
    rois = json.load(file)

data = []
labels = []

def load_and_label_images(folder, label):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Skipping file {file_path}")
            continue
        normalized_image = image / 255.0  # Normalize pixel values
        data.append(normalized_image)
        labels.append(label)

# Load data
load_and_label_images(good_dir, label=1)  # Good images -> Pass
load_and_label_images(bad_dir, label=0)   # Bad images -> Fail

# Convert to NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels, dtype="int")
print(f"Loaded {len(data)} images.")

# Step 2: Split Dataset
X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")

# Step 3: Augment Data (Optional)
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train.reshape(-1, X_train.shape[1], X_train.shape[2], 1))  # Add channel dimension

# Step 4: Save Preprocessed Data
with open("preprocessed_data.pkl", "wb") as f:
    pickle.dump((X_train, X_val, X_test, y_train, y_val, y_test), f)

print("Preprocessing complete. Data saved to 'preprocessed_data.pkl'.")
