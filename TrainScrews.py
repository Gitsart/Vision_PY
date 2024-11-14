import cv2
import os
import time

# Initialize video capture
cap = cv2.VideoCapture(0)  # Camera index; adjust if necessary

# Create directories for dataset if they don’t exist
if not os.path.exists("dataset/effective"):
    os.makedirs("dataset/effective")
if not os.path.exists("dataset/defective"):
    os.makedirs("dataset/defective")

effective_count = 0
defective_count = 0

print("Press 'e' to save an effective product image")
print("Press 'd' to save a defective product image")
print("Press 'q' to quit")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # Display the frame
    cv2.imshow("Live Feed - Press 'e' or 'd' to capture", frame)

    # Key press detection
    key = cv2.waitKey(1) & 0xFF

    if key == ord('e'):  # Save as effective
        effective_count += 1
        filename = f"dataset/effective/effective_{time.time()}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved effective product image as {filename}")

    elif key == ord('d'):  # Save as defective
        defective_count += 1
        filename = f"dataset/defective/defective_{time.time()}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved defective product image as {filename}")

    elif key == ord('q'):  # Quit
        print("Exiting and releasing camera...")
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
