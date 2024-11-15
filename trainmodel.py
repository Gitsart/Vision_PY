import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Load the data
with open("preprocessed_data.pkl", "rb") as f:
    data = pickle.load(f)

# Unpack the tuple into respective variables
X_train, X_val, X_test, y_train, y_val, y_test = data

# Normalize the pixel values to range [0, 1]
X_train, X_val, X_test = X_train / 255.0, X_val / 255.0, X_test / 255.0

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 256, 1)),  # Shape of image
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # 2 classes: good and bad
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32,
                    validation_data=(X_val, y_val))

# Save the model
model.save("screw_inspection_model")
print("Model training complete and saved to 'screw_inspection_model.h5'.")
