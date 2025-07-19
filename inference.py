import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder
import os

# Load model
model = load_model("emotion_model.h5")

# Load label encoder from training
emotion_labels = sorted([f.replace(".npy", "") for f in os.listdir("emotion_data") if f.endswith(".npy")])
le = LabelEncoder()
le.fit(emotion_labels)

# Load face detection model
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Start webcam
cap = cv2.VideoCapture(0)

print("📷 Starting webcam for emotion detection. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.80:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")
            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (48, 48))
            normalized = resized / 255.0
            reshaped = normalized.reshape(1, 48, 48, 1)

            pred = model.predict(reshaped)
            emotion_index = np.argmax(pred)
            emotion = le.inverse_transform([emotion_index])[0]

            # Draw
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{emotion} ({pred[0][emotion_index]*100:.1f}%)", 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                        (0, 255, 0), 2)

    cv2.imshow("Real-time Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
