import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import sys, os

# ---------------------------------------------------------
# Add project root so we can import from /models
# ---------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "models"))

from custom_cnn import CustomCNN

# ---------------------------------------------------------
# Device setup (CPU only)
# ---------------------------------------------------------
device = torch.device("cpu")

# ---------------------------------------------------------
# Emotion Labels (FER2013)
# ---------------------------------------------------------
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ---------------------------------------------------------
# Load trained model
# ---------------------------------------------------------
model = CustomCNN(num_classes=7)
model.load_state_dict(torch.load("Models/custom_cnn_cpu.pth", map_location=device))
model.to(device)
model.eval()

print("Model Loaded on CPU")

# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---------------------------------------------------------
# Load Haarcascade face detector
# ---------------------------------------------------------
cascade_path = os.path.join(CURRENT_DIR, "haarcascade_frontalface_default.xml")
print("Loading Haarcascade from:", cascade_path)

face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print("❌ ERROR: Haarcascade could not be loaded. Check file path.")
    sys.exit()

# ---------------------------------------------------------
# Initialize webcam
# ---------------------------------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ ERROR: Cannot access webcam")
    sys.exit()

print("Webcam is ON. Press 'q' to quit.")

# ---------------------------------------------------------
# Real-time loop
# ---------------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=4
    )

    for (x, y, w, h) in faces:
        # Crop face
        face_img = frame[y:y+h, x:x+w]

        # Convert to PIL and apply transforms
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)
        face_tensor = transform(face_pil).unsqueeze(0).to(device)

        # Predict emotion
        with torch.no_grad():
            output = model(face_tensor)
            _, pred = torch.max(output, 1)
            emotion = emotion_labels[pred.item()]

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

        # Display emotion label
        cv2.putText(
            frame,
            emotion,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Emotion Detection (Press Q to Quit)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
