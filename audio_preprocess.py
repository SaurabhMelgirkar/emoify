import os
import librosa
import numpy as np
import pandas as pd

# Specify the directory containing audio files
audio_directory = r"C:\Users\acer\Downloads"

# Function to extract audio features
def extract_features(file_path):
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=None)
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        
        # Extract Chroma Features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)
        
        # Extract Spectral Contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spectral_contrast_mean = np.mean(spectral_contrast.T, axis=0)
        
        # Combine features into a single array
        features = np.concatenate((mfccs_mean, chroma_mean, spectral_contrast_mean))
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Process all audio files in the specified directory
def preprocess_audio_files(directory):
    features = []
    for file_name in os.listdir(directory):
        if file_name.endswith(".wav") or file_name.endswith(".mp3"):  # Adjust file extensions if necessary
            file_path = os.path.join(directory, file_name)
            feature = extract_features(file_path)
            if feature is not None:
                features.append({
                    "file_name": file_name,
                    "features": feature
                })
    return features

# Extract features from audio files
audio_features = preprocess_audio_files(audio_directory)

# Convert the extracted features into a DataFrame for easy visualization
df = pd.DataFrame(audio_features)

# Display the first few rows of the DataFrame
print(df.head())

# Save the features to a CSV file for further analysis or model training
df.to_csv("audio_features.csv", index=False)
