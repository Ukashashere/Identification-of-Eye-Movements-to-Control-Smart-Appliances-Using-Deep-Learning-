#Accuracy is very slightli improved using our dataset, about 2000 imgs. ,kinda
#feels very very slightly better
# When eye is closed, it is definately closed
# Have to properly open the eye for 'open' status.

'''
I apologize for the oversight. It seems there's no direct property in OpenCV to minimize
a window. However, we can achieve similar functionality by resizing the window to a very
small size when the eye is closed. Let's adjust the code accordingly:

In this version, instead of minimizing the window, we resize it to a very small size (1x1) 
when the eye is closed, effectively hiding it. When the eye is open, we resize the window
back to its original size (640x480). This achieves a similar visual effect to minimizing 
the window.

'''

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import openvino as ov

# Load OpenVINO Core
core = ov.Core()

# Load OpenVINO Model
ir_model = core.read_model('blink_Ukasha/test1_blink_ukasha.xml')
compiled_model = core.compile_model(model=ir_model)
output_key = compiled_model.output(0)

# Load Keras Model
model = load_model('blink_Ukasha/test1_eye_ukasha.h5', compile=False)

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
            if probability_open > 0.03:
                eye_status = "Open"
                # Show the window if the eye is open
                cv2.resizeWindow('Eye Status Detection', 640, 480)
            else:
                eye_status = "Closed"
                # Minimize the window if the eye is closed
                cv2.resizeWindow('Eye Status Detection', 1, 1)

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