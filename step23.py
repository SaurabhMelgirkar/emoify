import pandas as pd
import numpy as np
from ast import literal_eval

# Load the dataset
csv_file = "audio_features.csv"
data = pd.read_csv(csv_file)

# Safely parse features from the "features" column
def safe_parse(feature_string):
    if pd.isna(feature_string):  # Check for NaN
        return None
    try:
        return np.array(literal_eval(feature_string))  # Parse array-like strings
    except Exception as e:
        print(f"Error parsing features: {e} | Data: {feature_string}")
        return None

# Apply the parsing function and clean the "features" column
data["features"] = data["features"].apply(safe_parse)
data = data.dropna(subset=["features"])  # Remove rows with invalid features

# Define a placeholder array (e.g., random noise or a default value)
placeholder_array = np.random.normal(loc=0, scale=1, size=32)  # Adjust size if needed

# Function to check if a feature array is all zeros
def is_all_zeros(feature_array):
    return np.all(feature_array == 0)

# Replace all-zero features with the placeholder array
data["features"] = data["features"].apply(lambda x: placeholder_array if is_all_zeros(x) else x)

# Verify the dataset
print(f"Number of rows after replacing all-zero features: {len(data)}")
print(data.head())  # Display a sample of the dataset

# Save the modified dataset back to a new CSV file
data.to_csv("updated_audio_features.csv", index=False)
print("Updated dataset saved to 'updated_audio_features.csv'")