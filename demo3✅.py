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
            else:
                eye_status = "Closed"

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


'''
This AIOT project integrates face and eye detection using Haarcascades,
preprocesses the detected eyes, and predicts their status (open or closed) using a 
combination of a Keras deep learning model and an OpenVINO optimized model. 
The application continuously captures video frames from the webcam and displays the 
processed frames in real-time.

For detecting the status of eyes: (open or closed) using a combination of 
OpenVINO (Open Visual Inference and Neural Network Optimization) and a Keras deep learning 
model. The project primarily involves face and eye detection using Haarcascades and then 
applying a pre-trained model for eye status prediction.

Let's break down the code step by step:

1. **Import Libraries:**
   ```python
   import cv2
   import numpy as np
   from tensorflow.keras.models import load_model
   import openvino as ov
   ```
   - `cv2`: OpenCV library for computer vision tasks.
   - `numpy`: Numerical computing library for handling arrays and matrices.
   - `load_model`: Function from TensorFlow Keras to load a pre-trained deep learning model.
   - `openvino`: OpenVINO toolkit for optimizing and deploying deep learning models.

2. **Load OpenVINO Core:**
   ```python
   core = ov.Core()
   ```
   - Creates an instance of the OpenVINO Core for handling model loading and inference.

3. **Load OpenVINO Model:**
   ```python
   ir_model = core.read_model('blink/blink_CNN.xml')
   compiled_model = core.compile_model(model=ir_model)
   output_key = compiled_model.output(0)
   ```
   - Reads an OpenVINO IR (Intermediate Representation) model file.
   - Compiles the model for efficient inference.
   - Defines the output key for accessing model predictions.

4. **Load Keras Model:**
   ```python
   model = load_model('blink/eye.h5', compile=False)
   ```
   - Loads a Keras deep learning model for further processing.

5. **Preprocess Eye Function:**
   ```python
   def preprocess_eye(eye):
       # Preprocesses the input eye image for prediction
   ```
   - Resizes the input eye image to (80, 80) pixels.
   - Normalizes pixel values to the range [0, 1].
   - Adds an extra dimension to the image array.

6. **Load Haarcascades for Face and Eyes Detection:**
   ```python
   face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
   eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
   ```
   - Loads Haarcascades XML files for face and eye detection.

7. **Open Webcam:**
   ```python
   cap = cv2.VideoCapture(0)
   ```
   - Opens the default camera (index 0) for capturing video frames.

8. **Main Loop:**
   ```python
   while True:
       # Capturing video frame
       ret, frame = cap.read()
       gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

       # Face detection
       faces = face_cascade.detectMultiScale(gray, minNeighbors=15, scaleFactor=1.2, 
       minSize=(25, 25))
   ```
   - Reads a frame from the webcam.
   - Converts the frame to grayscale.
   - Detects faces using Haarcascades.

9. **Face and Eye Processing:**
   ```python
       for (x, y, w, h) in faces:
           # ... (omitted for brevity)
           roi_gray = gray[y:y+h, x:x+w]
           roi_color = frame[y:y+h, x:x+w]

           # Eye detection
           eyes = eye_cascade.detectMultiScale(roi_gray)

           for (ex, ey, ew, eh) in eyes:
               # ... (omitted for brevity)
               eye_input = preprocess_eye(roi_color[ey:ey + eh, ex:ex + ew])

               # Eye status prediction using OpenVINO model
               prediction = compiled_model(eye_input)[output_key]
               probability_open = prediction[0][1]

               # Determine eye status
               if probability_open > 0.5:
                   eye_status = "Open"
               else:
                   eye_status = "Closed"

               # Display eye status on the frame
               cv2.putText(roi_color, f"Eye Status: {eye_status}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
   ```

10. **Display and Exit Handling:**
    ```python
       # Display the frame
       cv2.imshow('Eye Status Detection', frame)

       # Break the loop if 'q' key is pressed
       key = cv2.waitKey(1)
       if key in {ord('q'), ord('Q'), 27}:
               break
    ```
    - Displays the frame with detected face and eye regions.
    - Breaks the loop if the 'q' key or the 'Esc' key is pressed.

11. **Release Resources:**
    ```python
   # Release the camera and close all OpenCV windows
   cap.release()
   cv2.destroyAllWindows()
    ```
    - Releases the webcam and closes all OpenCV windows.

In summary, this AIOT project integrates face and eye detection using Haarcascades,
preprocesses the detected eyes, and predicts their status (open or closed) using a 
combination of a Keras deep learning model and an OpenVINO optimized model. 
The application continuously captures video frames from the webcam and displays the 
processed frames in real-time.
'''