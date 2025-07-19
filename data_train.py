import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Path where .npy files are saved
data_dir = "emotion_data"

X = []
y = []

# Load each emotion file
for file in os.listdir(data_dir):
    if file.endswith(".npy"):
        label = file.replace(".npy", "")
        data = np.load(os.path.join(data_dir, file))
        X.extend(data)
        y.extend([label] * len(data))

X = np.array(X)
y = np.array(y)

# Normalize pixel values (0–1)
X = X / 255.0

# Reshape to (samples, 48, 48, 1) for CNN input
X = X.reshape(-1, 48, 48, 1)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(np.unique(y)), activation='softmax')  # Number of emotion classes
])

model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))

# Save model
model.save("emotion_model.h5")
print("✅ Model trained and saved as 'emotion_model.h5'")
