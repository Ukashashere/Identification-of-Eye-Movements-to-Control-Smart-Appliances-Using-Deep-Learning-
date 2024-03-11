import cv2
import numpy as np
from tensorflow.keras.models import load_model
from pygame import mixer
import time
import openvino as ov

# Load OpenVINO Core
core = ov.Core()

# Load OpenVINO Model
ir_model = core.read_model('blink/blink_CNN.xml')
compiled_model = core.compile_model(model=ir_model)
output_key = compiled_model.output(0)

# Load Keras Model
model = load_model('blink/eye.h5', compile=False)

# Function to preprocess eye image
def preprocess_eye(eye):
    eye = cv2.resize(eye, (80, 80))
    eye = eye / 255.0
    eye = np.expand_dims(eye, axis=0)
    return eye

# Load Haarcascades for face and eyes detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, minNeighbors=15, scaleFactor=1.2, minSize=(25, 25))

    for (x, y, w, h) in faces:
        # Extract eye region
        eye = frame[y:y + h, x:x + w]
        # Preprocess eye for prediction
        eye_input = preprocess_eye(eye)

        # Run inference using OpenVINO model
        prediction = compiled_model(eye_input)[output_key]
        probability_open = prediction[0][1]
        probability_open_percent = probability_open * 100

        # Determine eye status
        if probability_open > 0.5:
            eye_status = "Open"
        else:
            eye_status = "Closed"

        # Display eye status on the frame
        cv2.putText(frame, f"Eye Status: {eye_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Display the frame
    cv2.imshow('Eye Status Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
