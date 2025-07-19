import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from ast import literal_eval

# Load the CSV file
csv_file = "audio_features.csv"
try:
    data = pd.read_csv(csv_file, on_bad_lines='skip')  # Skip problematic rows
except pd.errors.EmptyDataError:
    raise ValueError("The CSV file is empty or could not be read.")
except pd.errors.ParserError as e:
    raise ValueError(f"Error reading the CSV file: {e}")

# Check if the dataset is empty
if data.empty:
    raise ValueError("The dataset is empty. Please check your CSV file.")

print(f"Total rows in raw dataset: {len(data)}")

# Step 1: Safe parsing of "features" column
def safe_parse(feature_string):
    if pd.isna(feature_string):  # Handle missing values
        print("NaN detected; skipping this row.")
        return None
    try:
        return np.array(literal_eval(feature_string))  # Convert string to array
    except Exception as e:
        print(f"Parsing error: {e} | Skipping row: {feature_string}")
        return None

# Apply the parsing function to the "features" column
if "features" not in data.columns:
    raise ValueError("The 'features' column is missing in the dataset.")

data["features"] = data["features"].apply(safe_parse)

# Drop rows with missing or invalid features
data = data.dropna(subset=["features"])
print(f"Rows after parsing and dropping NaN: {len(data)}")

# Step 2: Handle all-zero feature rows
def is_all_zeros(feature_array):
    return np.all(feature_array == 0)

# Define a placeholder array for all-zero rows
placeholder_array = np.random.normal(loc=0, scale=1, size=32)  # Random array for fallback

# Replace all-zero features with placeholders
data["features"] = data["features"].apply(lambda x: placeholder_array if is_all_zeros(x) else x)

# Validate the dataset after cleaning
print(f"Rows remaining after replacing all-zero features: {len(data)}")
if len(data) == 0:
    raise ValueError("No valid feature data available after cleaning. Please check your dataset.")

# Step 3: Process feature data (X)
try:
    X = np.stack(data["features"].values)  # Convert features to 2D NumPy array
    print(f"Feature data shape (X): {X.shape}")
except ValueError as e:
    raise ValueError(f"Error stacking feature data: {e}")

# Step 4: Validate and process the "emotion" column
if "emotion" not in data.columns:
    raise ValueError("The 'emotion' column is missing. Please ensure your dataset includes labels.")

y = data["emotion"]
if y.isnull().all():
    raise ValueError("The 'emotion' column contains no valid labels. Please check your dataset.")

# Encode emotion labels into numerical format
label_encoder = LabelEncoder()
try:
    y = label_encoder.fit_transform(y)
except Exception as e:
    raise ValueError(f"Error encoding emotion labels: {e}")

# Step 5: Split data into training and testing sets
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")
except ValueError as e:
    raise ValueError(f"Error during train-test split: {e}")

# Step 6: Normalize feature data
try:
    X_train = X_train / np.max(X_train)
    X_test = X_test / np.max(X_test)
except Exception as e:
    raise ValueError(f"Error normalizing feature data: {e}")

# Step 7: Define the model architecture
try:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(np.unique(y)), activation='softmax')  # Output layer for emotions
    ])
except Exception as e:
    raise ValueError(f"Error defining the model architecture: {e}")

# Compile the model
try:
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print("Model compiled successfully.")
except Exception as e:
    raise ValueError(f"Error compiling the model: {e}")

# Train the model
try:
    history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))
    print("Model training completed.")
except Exception as e:
    raise ValueError(f"Error training the model: {e}")

# Evaluate the model
try:
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
except Exception as e:
    raise ValueError(f"Error evaluating the model: {e}")

# Save the trained model
try:
    model.save("emotion_detection_model.h5")
    print("Model saved successfully.")
except Exception as e:
    raise ValueError(f"Error saving the model: {e}")

# Map emotions back to their label names
try:
    emotion_labels = label_encoder.classes_
    print("Emotion Labels:", emotion_labels)
except Exception as e:
    raise ValueError(f"Error mapping labels: {e}")