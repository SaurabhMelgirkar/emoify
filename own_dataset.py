import librosa
import pandas as pd
import numpy as np

# Initialize empty dataset
audio_features = []

# List of audio files to process
file_names = ["audio_1.wav", "audio_2.wav"]  # Replace with your file paths

# Extract features from each audio file
for file_name in file_names:
    try:
        y, sr = librosa.load(file_name, sr=None)  # Load audio
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=32)  # Extract 32 MFCCs
        mfccs_mean = mfccs.mean(axis=1)  # Average MFCC values
        audio_features.append({"file_name": file_name, "features": mfccs_mean.tolist(), "emotion": "happy"})  # Add label
    except Exception as e:
        print(f"Error processing {file_name}: {e}")

# Save as CSV
df = pd.DataFrame(audio_features)
df.to_csv("generated_audio_features.csv", index=False)
print("Dataset saved to 'generated_audio_features.csv'")