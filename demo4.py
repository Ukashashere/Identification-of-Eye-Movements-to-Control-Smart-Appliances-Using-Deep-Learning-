import cv2
import numpy as np
from tensorflow.keras.models import load_model
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

# Create a resizable window
cv2.namedWindow('Eye Status Detection', cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, minNeighbors=15, scaleFactor=1.2, minSize=(25, 25))

    for (x, y, w, h) in faces:
        # Draw rectangular box around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Extract face region
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes in the face region
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            # Draw rectangular box around the detected eyes
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

            # Preprocess eye for prediction
            eye_input = preprocess_eye(roi_color[ey:ey + eh, ex:ex + ew])

            # Run inference using OpenVINO model
            prediction = compiled_model(eye_input)[output_key]
            probability_open = prediction[0][1]
            probability_open_percent = probability_open * 100

            # Determine eye status
            if probability_open > 0.5:
                eye_status = "Open"
                # Show the window if the eye is open
                cv2.setWindowProperty('Eye Status Detection', cv2.WND_PROP_VISIBLE, cv2.WINDOW_NORMAL)
            else:
                eye_status = "Closed"
                # Hide the window if the eye is closed
                cv2.setWindowProperty('Eye Status Detection', cv2.WND_PROP_VISIBLE, cv2.WINDOW_MINIMIZED)

            # Display eye status on the frame
            cv2.putText(roi_color, f"Eye Status: {eye_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Eye Status Detection', frame)

    # Break the loop if 'q' key is pressed
    key = cv2.waitKey(1)
    if key in {ord('q'), ord('Q'), 27}:
            break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
