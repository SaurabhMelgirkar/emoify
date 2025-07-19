# import cv2
# import numpy as np
# import os
# import time

# # Load DNN face detector
# modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
# configFile = "deploy.prototxt.txt"
# net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# # Emotion label input
# emotion_label = input("Enter the emotion label (e.g., happy, sad, angry): ").strip().lower()

# # Create save directory
# save_dir = "emotion_data"
# os.makedirs(save_dir, exist_ok=True)

# # Webcam init
# cap = cv2.VideoCapture(0)
# max_images = 100
# capture_count = 0
# data = []

# print(f"Starting capture for emotion: '{emotion_label}'")
# time.sleep(2)  # Wait a bit before starting

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     h, w = frame.shape[:2]
#     blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 
#                                  1.0, (300, 300), (104.0, 177.0, 123.0))
#     net.setInput(blob)
#     detections = net.forward()

#     best_face = None
#     best_confidence = 0

#     for i in range(detections.shape[2]):
#         confidence = detections[0, 0, i, 2]

#         if confidence > 0.80 and confidence > best_confidence:
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (x1, y1, x2, y2) = box.astype("int")
#             best_face = frame[y1:y2, x1:x2]
#             best_confidence = confidence
#             best_box = (x1, y1, x2, y2)

#     if best_face is not None and best_face.size != 0:
#         gray_face = cv2.cvtColor(best_face, cv2.COLOR_BGR2GRAY)
#         resized_face = cv2.resize(gray_face, (48, 48))
#         data.append(resized_face)
#         capture_count += 1

#         x1, y1, x2, y2 = best_box
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(frame, f"Capturing {capture_count}/{max_images}", (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#         # Wait briefly between captures (to avoid rapid duplicates)
#         cv2.imshow("Capturing Faces", frame)
#         cv2.waitKey(150)  # delay in milliseconds

#     else:
#         cv2.putText(frame, "No high-confidence face detected", (20, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#     cv2.imshow("Capturing Faces", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q') or capture_count >= max_images:
#         break

# cap.release()
# cv2.destroyAllWindows()

# # Save the data
# data = np.array(data)
# np.save(os.path.join(save_dir, f"{emotion_label}.npy"), data)
# print(f"\n✅ Saved {len(data)} images to '{save_dir}/{emotion_label}.npy'")


import cv2
import numpy as np
import os
import time
import sys

# Get emotion label from command-line argument
if len(sys.argv) < 2:
    print("❌ No emotion label provided.")
    sys.exit(1)

emotion_label = sys.argv[1].strip().lower()

# Load DNN face detector
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Create save directory
save_dir = "emotion_data"
os.makedirs(save_dir, exist_ok=True)

# Webcam init
cap = cv2.VideoCapture(0)
max_images = 100
capture_count = 0
data = []

print(f"Starting capture for emotion: '{emotion_label}'")
time.sleep(2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    best_face = None
    best_confidence = 0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.80 and confidence > best_confidence:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            best_face = frame[y1:y2, x1:x2]
            best_confidence = confidence
            best_box = (x1, y1, x2, y2)

    if best_face is not None and best_face.size != 0:
        gray_face = cv2.cvtColor(best_face, cv2.COLOR_BGR2GRAY)
        resized_face = cv2.resize(gray_face, (48, 48))
        data.append(resized_face)
        capture_count += 1

        x1, y1, x2, y2 = best_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Capturing {capture_count}/{max_images}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Capturing Faces", frame)
        cv2.waitKey(150)

    else:
        cv2.putText(frame, "No high-confidence face detected", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Capturing Faces", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or capture_count >= max_images:
        break

cap.release()
cv2.destroyAllWindows()

# Save the data
data = np.array(data)
np.save(os.path.join(save_dir, f"{emotion_label}.npy"), data)
print(f"\n✅ Saved {len(data)} images to '{save_dir}/{emotion_label}.npy'")

