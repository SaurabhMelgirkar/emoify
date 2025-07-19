import pandas as pd
import numpy as np

# Load the CSV file
csv_file = "audio_features.csv"
data = pd.read_csv(csv_file)

# Safely parse features from the "features" column
def safe_parse(feature_string):
    from ast import literal_eval
    if pd.isna(feature_string):  # Check for missing values (NaN)
        return None
    try:
        return np.array(literal_eval(feature_string))  # Parse array-like strings
    except Exception as e:
        print(f"Error parsing features: {e} | Data: {feature_string}")
        return None

# Apply the parsing function and clean the "features" column
data["features"] = data["features"].apply(safe_parse)
data = data.dropna(subset=["features"])  # Remove rows with invalid features

# Function to check if an array is all zeros
def is_all_zeros(feature_array):
    return np.all(feature_array == 0)

# Filter out rows with all-zero features
data = data[~data["features"].apply(is_all_zeros)]

# Verify the number of remaining valid rows
print(f"Number of valid rows after removing all-zero features: {len(data)}")
if len(data) == 0:
    raise ValueError("No valid feature data left after removing all-zero features.")