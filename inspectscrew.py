import cv2
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('screw_inspection_model')

# Capture video (from camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize the frame to match the model's input shape (32, 256)
    gray_frame = cv2.resize(gray_frame, (256, 32))

    # Reshape to match the expected input shape (batch_size, height, width, channels)
    gray_frame = gray_frame.reshape((1, 32, 256, 1))  # 1 channel for grayscale

    # Normalize the image (if required by your model)
    gray_frame = gray_frame / 255.0

    # Make predictions
    predictions = model.predict(gray_frame)

    # Display prediction or results
    print(predictions)

    # Show the original frame for visualization
    cv2.imshow('Frame', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
