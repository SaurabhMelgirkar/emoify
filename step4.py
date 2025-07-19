import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from ast import literal_eval

# Load the CSV file
csv_file = "audio_features.csv"  # Ensure your CSV file is in the same directory
data = pd.read_csv(csv_file)

# Check if the dataset is empty
if data.empty:
    raise ValueError("The dataset is empty. Please check your CSV file.")

print(f"Dataset loaded. Total rows: {len(data)}")

# Parse the "features" column
def safe_parse(feature_string):
    try:
        return np.array(literal_eval(feature_string))  # Convert string to NumPy array
    except Exception as e:
        print(f"Error parsing features: {e} | Data: {feature_string}")
        return None

# Apply parsing to the "features" column
data["features"] = data["features"].apply(safe_parse)

# Drop rows with invalid or missing features
data = data.dropna(subset=["features"])
print(f"Rows remaining after parsing: {len(data)}")

# Ensure the "emotion" column exists
if "emotion" not in data.columns:
    raise ValueError("The 'emotion' column is missing. Please ensure your dataset includes labels.")

# Extract features (X) and labels (y)
X = np.stack(data["features"].values)  # Convert features to 2D NumPy array
y = data["emotion"]

# Encode labels (emotions) into integers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the feature data
X_train = X_train / np.max(X_train)
X_test = X_test / np.max(X_test)

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),  # Input layer
    tf.keras.layers.Dropout(0.3),  # Dropout for regularization
    tf.keras.layers.Dense(64, activation='relu'),  # Hidden layer
    tf.keras.layers.Dropout(0.3),  # Dropout for regularization
    tf.keras.layers.Dense(len(np.unique(y)), activation='softmax')  # Output layer for classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=30,
                    batch_size=32)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Save the model as model3.h5
model.save("model3.h5")
print("Model saved as 'model3.h5'.")

# Map emotions back to their original labels
emotion_labels = label_encoder.classes_
print("Emotion Labels:", emotion_labels)