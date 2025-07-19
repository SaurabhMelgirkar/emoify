import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
from sklearn.model_selection import train_test_split

# Step 1: Generate Dummy Data (Replace with Your Actual Data)
# Assuming the input data has 32 features and output has 3 classes
num_samples = 1000  # Number of samples in the dummy dataset
num_features = 32   # Number of features per input sample
num_classes = 3     # Number of output classes (e.g., emotions)

# Create dummy features (X) and labels (y)
X = np.random.random((num_samples, num_features))  # Random feature values
y = np.random.randint(num_classes, size=num_samples)  # Random integer labels

# Step 2: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Normalize Feature Data
X_train = X_train / np.max(X_train)  # Scale features to range 0-1
X_test = X_test / np.max(X_test)

# Step 4: Build the Model
model = Sequential([
    Dense(128, activation='relu', input_shape=(num_features,)),  # Input layer
    Dropout(0.3),  # Dropout for regularization
    Dense(64, activation='relu'),  # Hidden layer
    Dropout(0.3),  # Dropout for regularization
    Dense(num_classes, activation='softmax')  # Output layer with softmax for classification
])

# Step 5: Compile the Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 6: Train the Model
history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=30,
                    batch_size=32)

# Step 7: Evaluate the Model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Step 8: Save the Model as model2.h5
model.save("model2.h5")
print("Model saved as 'model2.h5'")