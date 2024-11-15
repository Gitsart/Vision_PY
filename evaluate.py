import tensorflow as tf
import pickle

# Load the trained model
model = tf.keras.models.load_model("screw_inspection_model.h5")

# Load the preprocessed data
with open("preprocessed_data.pkl", "rb") as f:
    data = pickle.load(f)

# Unpack the data
X_train, X_val, X_test, y_train, y_val, y_test = data

# Normalize the test set (same preprocessing as during training)
X_test = X_test / 255.0

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_accuracy}")
