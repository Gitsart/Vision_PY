import cv2
import os
import time
import subprocess

# Create directories for dataset if they don’t exist
if not os.path.exists("dataset/effective"):
    os.makedirs("dataset/effective")
if not os.path.exists("dataset/defective"):
    os.makedirs("dataset/defective")

effective_count = 0
defective_count = 0

print("Press 'e' to save effective product image")
print("Press 'd' to save defective product image")
print("Press 'q' to quit")

while True:
    # Display a placeholder message for live view simulation
    print("Live view simulation - press 'e' or 'd' to capture")

    # Key press detection
    key = input("Press key: ")

    if key == 'e':  # Save as effective
        effective_count += 1
        filename = f"dataset/effective/effective_{time.time()}.jpg"
        subprocess.run(["libcamera-still", "-o", filename])
        print(f"Saved effective product image as {filename}")

    elif key == 'd':  # Save as defective
        defective_count += 1
        filename = f"dataset/defective/defective_{time.time()}.jpg"
        subprocess.run(["libcamera-still", "-o", filename])
        print(f"Saved defective product image as {filename}")

    elif key == 'q':  # Quit
        print("Exiting...")
        break
