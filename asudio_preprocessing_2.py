import os
import librosa
import pandas as pd
import numpy as np

# Define the path to the dataset folder
dataset_folder = r"C:\Users\acer\Downloads"

# Collect all .wav files in the dataset folder
audio_files = [os.path.join(dataset_folder, file) for file in os.listdir(dataset_folder) if file.endswith('.wav')]

# Example: Assign labels dynamically or use a default placeholder if labels are missing
labels = ["happy", "sad", "neutral", "angry", "calm"]  # Update this to reflect your use case
if len(audio_files) > len(labels):
    # Add 'unknown' as placeholders for files without labels
    labels += ["unknown"] * (len(audio_files) - len(labels))

# Ensure the number of audio files and labels match
if len(audio_files) != len(labels):
    print(f"Number of audio files: {len(audio_files)}")
    print(f"Number of labels: {len(labels)}")
    raise ValueError("The number of audio files and labels do not match. Please ensure they align correctly.")

# Initialize an empty list to store the processed data
data = []

# Extract features and associate them with labels
for file_name, emotion in zip(audio_files, labels):
    try:
        # Load the audio file
        y, sr = librosa.load(file_name, sr=None)
        
        # Extract 32 MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=32).mean(axis=1)
        
        # Append the data to the list
        data.append({"file_name": os.path.basename(file_name), "features": mfccs.tolist(), "emotion": emotion})
    except Exception as e:
        print(f"Error processing {file_name}: {e}")

# Check if any data was successfully processed
if len(data) == 0:
    raise ValueError("No audio files were successfully processed. Please check the dataset and audio file formats.")

# Save the dataset to a CSV file
output_csv = os.path.join(dataset_folder, "audio_features.csv")
df = pd.DataFrame(data)
df.to_csv(output_csv, index=False)
print(f"Dataset saved as 'audio_features.csv' at: {output_csv}")