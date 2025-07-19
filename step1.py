# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from ast import literal_eval

# # Load the dataset
# csv_file = "audio_features.csv"
# data = pd.read_csv(csv_file)

# # Check if the dataset is empty
# if data.empty:
#     raise ValueError("The dataset is empty. Please check your CSV file.")

# # Debug: Print raw data
# print(f"Dataset loaded. Total rows: {len(data)}")

# # Function to safely parse features
# def safe_parse(feature_string):
#     if pd.isna(feature_string):  # Handle NaN
#         print("NaN detected; skipping this row.")
#         return None
#     try:
#         return np.array(literal_eval(feature_string))  # Convert to NumPy array
#     except Exception as e:
#         print(f"Error parsing features: {e} | Data: {feature_string}")
#         return None

# # Apply feature parsing
# data["features"] = data["features"].apply(safe_parse)

# # Drop rows with invalid or missing features
# data = data.dropna(subset=["features"])

# # Debug: Check the number of valid rows after parsing
# print(f"Rows remaining after parsing: {len(data)}")

# # Handle all-zero rows by replacing them with placeholders
# def is_all_zeros(feature_array):
#     return np.all(feature_array == 0)

# placeholder_array = np.random.normal(loc=0, scale=1, size=32)  # Example random features
# data["features"] = data["features"].apply(lambda x: placeholder_array if is_all_zeros(x) else x)

# # Ensure there are valid rows
# print(f"Rows remaining after cleaning: {len(data)}")
# if len(data) == 0:
#     raise ValueError("No valid feature data available after cleaning. Please check your dataset.")

# # Prepare feature matrix (X) and labels (y)
# X = np.stack(data["features"].values)
# print(f"Shape of feature data: {X.shape}")

# if "emotion" not in data.columns:
#     raise ValueError("The 'emotion' column is missing. Please ensure your dataset includes labels.")

# y = data["emotion"]
# if y.isnull().all():
#     raise ValueError("The 'emotion' column contains no valid labels. Please check your dataset.")

# # Encode labels
# label_encoder = LabelEncoder()
# y = label_encoder.fit_transform(y)

# # Split data into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Normalize feature data
# X_train = X_train / np.max(X_train)
# X_test = X_test / np.max(X_test)

# # Define model architecture
# model = tf.keras.Sequential([
#     tf.keras.layers.Input(shape=(X_train.shape[1],)),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dropout(0.3),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dropout(0.3),
#     tf.keras.layers.Dense(len(np.unique(y)), activation='softmax')  # Output layer
# ])

# # Compile and train the model
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# # Evaluate the model
# test_loss, test_accuracy = model.evaluate(X_test, y_test)
# print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# # Save the model
# model.save("emotion_detection_model.h5")
# print("Model saved as 'emotion_detection_model.h5'.")

# # Map labels to their corresponding emotions
# emotion_labels = label_encoder.classes_
# print("Emotion Labels:", emotion_labels)

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from ast import literal_eval

# Load the dataset
csv_file = "audio_features.csv"
data = pd.read_csv(csv_file)

# Check if the dataset is empty
if data.empty:
    raise ValueError("The dataset is empty. Please check your CSV file.")

# Debug: Print raw data
print(f"Dataset loaded. Total rows: {len(data)}")

# Function to safely parse features
def safe_parse(feature_string):
    try:
        return np.array(literal_eval(feature_string))  # Convert to NumPy array
    except Exception as e:
        print(f"Error parsing features: {e} | Data: {feature_string}")
        return None

# Apply feature parsing
data["features"] = data["features"].apply(safe_parse)

# Debug: Check the number of valid rows after parsing
print(f"Rows remaining after parsing: {len(data)}")

# Handle all-zero rows by replacing them with placeholders
def is_all_zeros(feature_array):
    return feature_array is not None and np.all(feature_array == 0)

placeholder_array = np.random.normal(loc=0, scale=1, size=32)  # Example random features
data["features"] = data["features"].apply(lambda x: placeholder_array if is_all_zeros(x) else x)

# Ensure there are valid rows
if len(data) == 0 or data["features"].isnull().all():
    raise ValueError("No valid feature data available. Please check your dataset.")

# Prepare feature matrix (X) and labels (y)
try:
    X = np.stack(data["features"].values)
    print(f"Shape of feature data: {X.shape}")
except ValueError as e:
    raise ValueError(f"Error preparing feature matrix: {e}")

# Ensure 'emotion' column is present and valid
if "emotion" not in data.columns:
    raise ValueError("The 'emotion' column is missing. Please ensure your dataset includes labels.")

y = data["emotion"]
if y.isnull().all():
    raise ValueError("The 'emotion' column contains no valid labels. Please check your dataset.")

# Encode labels
label_encoder = LabelEncoder()
try:
    y = label_encoder.fit_transform(y)
except Exception as e:
    raise ValueError(f"Error encoding labels: {e}")

# Split data into train and test sets
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")
except ValueError as e:
    raise ValueError(f"Error during train-test split: {e}")

# Normalize feature data
X_train = X_train / np.max(X_train)
X_test = X_test / np.max(X_test)

# Define model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(np.unique(y)), activation='softmax')  # Output layer
])

# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Save the model
model.save("emotion_detection_model.h5")
print("Model saved as 'emotion_detection_model.h5'.")

# Map labels to their corresponding emotions
emotion_labels = label_encoder.classes_
print("Emotion Labels:", emotion_labels)