1. **Image-based Examples**

   1. Detect faces in a single image
   2. Draw bounding boxes around faces
   3. Encode known faces & print encodings
   4. Recognize known faces in an image
   5. Calculate face distances between encodings
   6. Detect facial landmarks (eyes, nose, mouth)
   7. Extract & save cropped face images
   8. Face clustering (group similar faces)
   9. Rotate faces to upright position
   10. Detect face orientation (left, right, up, down)

2. **Video-based Examples**
   11\. Detect faces frame by frame in a video file
   12\. Recognize known faces in a video file
   13\. Real-time webcam face detection
   14\. Real-time webcam face recognition
   15\. Simple attendance logger (with timestamps)
   16\. Face blurring/anonymization in video
   17\. Real-time video frame rate (FPS) measurement
   18\. Facial expression detection (smiling, etc.)
   19\. Face swapping demo (advanced)
   20\. Creating a face recognition attendance system

3. **Advanced / Extras**
   21\. Performance measurement (FPS counter)
   22\. Encoding database persistence (save/load `.npy`)
   23\. Flask integration for a web demo
   24\. Emotion detection using landmarks (e.g., smile vs. frown)
   25\. Face mask detection with OpenCV and `face_recognition`
   26\. Age and gender prediction using landmarks
   27\. Using `face_recognition` with OpenCV for multi-face tracking
   28\. Use with TensorFlow for deep learning integration
   29\. Face detection using Haar cascades for comparison
   30\. Use in augmented reality (AR) applications
   31\. Recognizing faces in group photos (multi-person)
   32\. Live face recognition using Raspberry Pi
   33\. Cloud integration for face recognition storage
   34\. Social media post tagging using face recognition
   35\. Using facial recognition for personal security systems

---

### 1. **Detect Faces in a Single Image**

```python
import face_recognition
import cv2
from matplotlib import pyplot as plt

# Load image
image = face_recognition.load_image_file("input.jpg")

# Detect face locations
face_locations = face_recognition.face_locations(image)
print(f"Found {len(face_locations)} face(s) in this image.")

# Draw boxes around faces
for top, right, bottom, left in face_locations:
    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

# Show the result
plt.imshow(image)
plt.axis('off')
plt.show()
```

---

### 2. **Draw Bounding Boxes Around Faces**

```python
import face_recognition, cv2

def draw_boxes(image_path, output_path):
    img = face_recognition.load_image_file(image_path)
    locations = face_recognition.face_locations(img)
    for top, right, bottom, left in locations:
        cv2.rectangle(img, (left, top), (right, bottom), (0,255,0), 3)
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

draw_boxes("input.jpg", "boxed.jpg")
print("Saved boxed.jpg")
```

---

### 3. **Encode Known Faces & Print Encodings**

```python
import face_recognition

known_images = ["alice.jpg", "bob.jpg", "carol.jpg"]
encodings = []

for img_path in known_images:
    img = face_recognition.load_image_file(img_path)
    enc = face_recognition.face_encodings(img)
    if enc:
        print(f"{img_path} → encoding length: {len(enc[0])}")
        encodings.append(enc[0])
    else:
        print(f"No face found in {img_path}")
```

---

### 4. **Recognize Known Faces in an Image**

```python
import face_recognition, cv2

# Load known data
known_imgs = ["alice.jpg", "bob.jpg"]
known_names = ["Alice", "Bob"]
known_encs = [face_recognition.face_encodings(face_recognition.load_image_file(i))[0]
              for i in known_imgs]

# Load target
img = face_recognition.load_image_file("group.jpg")
locs = face_recognition.face_locations(img)
encs = face_recognition.face_encodings(img, locs)

for (top, right, bottom, left), face_enc in zip(locs, encs):
    matches = face_recognition.compare_faces(known_encs, face_enc)
    name = known_names[matches.index(True)] if True in matches else "Unknown"
    cv2.rectangle(img, (left, top), (right, bottom), (255,0,0), 2)
    cv2.putText(img, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

# Display with matplotlib
from matplotlib import pyplot as plt
plt.imshow(img); plt.axis('off'); plt.show()
```

---

### 5. **Calculate Face Distances**

```python
import face_recognition

# Assume 'known_enc' and 'unknown_enc' exist
distances = face_recognition.face_distance([known_enc], unknown_enc)
print(f"Distance: {distances[0]:.4f}")
# Closer to 0 = more similar
```

---

### 6. **Detect Facial Landmarks**

```python
import face_recognition

img = face_recognition.load_image_file("face.jpg")
landmarks = face_recognition.face_landmarks(img)[0]
print("Left eye points:", landmarks['left_eye'])
print("Top lip points:", landmarks['top_lip'])
```

---

### 7. **Extract & Save Cropped Faces**

```python
import face_recognition, cv2

img = face_recognition.load_image_file("group.jpg")
locs = face_recognition.face_locations(img)
for i, (t, r, b, l) in enumerate(locs):
    face = img[t:b, l:r]
    cv2.imwrite(f"face_{i}.jpg", cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
```

---

### 8. **Face Clustering (Group Similar Faces)**

```python
from sklearn.cluster import DBSCAN
import face_recognition, numpy as np

# Load several images → get all encodings in list 'all_encs'
# ...
model = DBSCAN(metric="euclidean", eps=0.5, min_samples=2)
labels = model.fit_predict(np.array(all_encs))
print("Cluster labels:", labels)
```

---

### 9. **Detect Faces in Video File**

```python
import cv2, face_recognition

video = cv2.VideoCapture("video.mp4")
while True:
    ret, frame = video.read()
    if not ret: break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locs = face_recognition.face_locations(rgb)
    for t,r,b,l in locs:
        cv2.rectangle(frame, (l,t),(r,b),(0,255,0),2)
    cv2.imshow("Video Faces", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release(); cv2.destroyAllWindows()
```

---

### 10. **Recognize Known Faces in Video File**

```python
import cv2, face_recognition

# Pre‑load known_encs & known_names
# ...
cap = cv2.VideoCapture("video.mp4")
while True:
    ret, f = cap.read()
    if not ret: break
    rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    locs = face_recognition.face_locations(rgb)
    encs = face_recognition.face_encodings(rgb, locs)
    for (t,r,b,l), enc in zip(locs, encs):
        matches = face_recognition.compare_faces(known_encs, enc)
        name = known_names[matches.index(True)] if True in matches else "Unknown"
        cv2.rectangle(f,(l,t),(r,b),(255,0,0),2)
        cv2.putText(f,name,(l,t-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)
    cv2.imshow("Recognize Video", f)
    if cv2.waitKey(1)==ord('q'): break
cap.release(); cv2.destroyAllWindows()
```

---

### 11. **Real‑time Webcam Face Detection**

```python
import cv2, face_recognition

cam = cv2.VideoCapture(0)
while True:
    _, frame = cam.read()
    small = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    locs = face_recognition.face_locations(rgb)
    for t,r,b,l in locs:
        cv2.rectangle(frame, (l*2,t*2),(r*2,b*2),(0,255,0),2)
    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1)==ord('q'): break
cam.release(); cv2.destroyAllWindows()
```

---

### 12. **Real‑time Webcam Face Recognition**

```python
import cv2, face_recognition

# Load known_encs & names here
# ...
cam = cv2.VideoCapture(0)
while True:
    _, frame = cam.read()
    small = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    locs = face_recognition.face_locations(rgb)
    encs = face_recognition.face_encodings(rgb, locs)
    for (t,r,b,l), enc in zip(locs, encs):
        matches = face_recognition.compare_faces(known_encs, enc)
        name = known_names[matches.index(True)] if True in matches else "Unknown"
        # scale coords back
        cv2.rectangle(frame, (l*4,t*4),(r*4,b*4),(255,0,0),2)
        cv2.putText(frame,name,(l*4,t*4-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)
    cv2.imshow("Webcam Recognizer", frame)
    if cv2.waitKey(1)==ord('q'): break
cam.release(); cv2.destroyAllWindows()
```

---

### 13. **Simple Attendance Logger**

```python
import cv2, face_recognition, csv, datetime

known_encs, known_names = load_known()  # as before
attendance_file = open("attendance.csv","w",newline="")
writer = csv.writer(attendance_file)
writer.writerow(["Name","Time"])

cam = cv2.VideoCapture(0)
seen = set()
while True:
    _, frame = cam.read()
    small = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    locs = face_recognition.face_locations(rgb)
    for enc in face_recognition.face_encodings(rgb, locs):
        matches = face_recognition.compare_faces(known_encs, enc)
        if True in matches:
            name = known_names[matches.index(True)]
            if name not in seen:
                seen.add(name)
                writer.writerow([name, datetime.datetime.now()])
                print(f"Marked {name}")
    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1)==ord('q'): break

attendance_file.close()
cam.release(); cv2.destroyAllWindows()
```

---

### 14. **Face Blurring / Anonymization in Video**

```python
import cv2, face_recognition

cap = cv2.VideoCapture("video.mp4")
while True:
    ret, f = cap.read()
    if not ret: break
    rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    locs = face_recognition.face_locations(rgb)
    for t,r,b,l in locs:
        face = f[t:b, l:r]
        blurred = cv2.GaussianBlur(face, (99,99), 30)
        f[t:b, l:r] = blurred
    cv2.imshow("Blurred Faces", f)
    if cv2.waitKey(1)==ord('q'): break
cap.release(); cv2.destroyAllWindows()
```

---

### 16. **Real-time Video Frame Rate (FPS) Measurement**

```python
import cv2
import time

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize FPS counter
fps = 0
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Display FPS on video frame
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow("Real-time FPS", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

### 17. **Facial Expression Detection (Smiling, etc.)**

```python
import cv2
import face_recognition

# Load a sample image with facial landmarks
image = face_recognition.load_image_file("smiling_face.jpg")
landmarks = face_recognition.face_landmarks(image)

# Check for smiling expression based on mouth landmarks
if landmarks:
    top_lip = landmarks[0]["top_lip"]
    bottom_lip = landmarks[0]["bottom_lip"]
    distance = sum([abs(top[1] - bottom[1]) for top, bottom in zip(top_lip, bottom_lip)])
    
    if distance > 20:  # Threshold distance for a smile
        print("Smiling detected!")
    else:
        print("No smile detected")
else:
    print("No face detected")
```

---

### 18. **Face Mask Detection (Using Face Recognition)**

```python
import face_recognition
import cv2

# Load image with face mask
image = face_recognition.load_image_file("masked_face.jpg")
face_locations = face_recognition.face_locations(image)

# Assuming the presence of a mask is determined by detecting the lower portion of the face
for (top, right, bottom, left) in face_locations:
    face_image = image[top:bottom, left:right]
    # If the nose is absent in the face region, assume mask detection
    nose = face_recognition.face_landmarks(face_image)[0].get('nose_bridge')
    if nose:
        print("Face mask detected.")
    else:
        print("No face mask detected.")
```

---

### 19. **Face Swapping Using Dlib (Advanced)**

```python
import dlib
import face_recognition
import cv2

# Load the images
image1 = cv2.imread("face1.jpg")
image2 = cv2.imread("face2.jpg")

# Convert to RGB for face recognition
rgb1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
rgb2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

# Get face locations and landmarks
face_locations1 = face_recognition.face_locations(rgb1)
face_locations2 = face_recognition.face_locations(rgb2)

landmarks1 = face_recognition.face_landmarks(rgb1, face_locations1)
landmarks2 = face_recognition.face_landmarks(rgb2, face_locations2)

# Implement swapping logic with dlib
# Note: Complex logic needed for face swapping, including affine transformations, which can be done using OpenCV and dlib
```

---

### 20. **Face Recognition Attendance System**

```python
import cv2
import face_recognition
import csv
import datetime

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load known faces and their encodings
known_faces = ["alice.jpg", "bob.jpg"]
known_names = ["Alice", "Bob"]
known_encodings = [face_recognition.face_encodings(face_recognition.load_image_file(img))[0] for img in known_faces]

# Open a CSV file to save attendance
attendance_file = open("attendance.csv", "w", newline="")
writer = csv.writer(attendance_file)
writer.writerow(["Name", "Time"])

# Set of recognized people
recognized_people = set()

while True:
    ret, frame = cap.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]

        # Record the name and time if not already recognized
        if name != "Unknown" and name not in recognized_people:
            recognized_people.add(name)
            writer.writerow([name, datetime.datetime.now()])
            print(f"Attendance marked for: {name}")

        # Draw bounding box and name
        for (top, right, bottom, left), name in zip(face_locations, [name] * len(face_locations)):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show frame with face recognition
    cv2.imshow("Face Recognition Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

attendance_file.close()
cap.release()
cv2.destroyAllWindows()
```

---

### 21. **Face Mask Detection with Deep Learning Model Integration**

```python
import cv2
import face_recognition
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import MobileNetV2

# Load pre-trained MobileNetV2 for mask detection
model = MobileNetV2(weights='imagenet')

# Load the image
img = cv2.imread("masked_face.jpg")

# Convert to RGB
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Detect faces in the image
face_locations = face_recognition.face_locations(rgb_img)

for (top, right, bottom, left) in face_locations:
    # Crop face
    face = img[top:bottom, left:right]
    
    # Preprocess face image
    face = cv2.resize(face, (224, 224))
    face_array = image.img_to_array(face)
    face_array = face_array / 255.0
    face_array = face_array.reshape((1, 224, 224, 3))
    
    # Predict mask vs. no-mask
    prediction = model.predict(face_array)
    print("Prediction:", prediction)
```

---

### 22. **Face Tracking in Video (Multi-Face Tracking)**

```python
import cv2
import face_recognition

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect all faces in the frame
    face_locations = face_recognition.face_locations(rgb_frame)

    # Draw bounding boxes around each face
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    cv2.imshow("Face Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

### 23. **Emotion Detection Using Landmarks (Smile vs. Frown)**

```python
import face_recognition

image = face_recognition.load_image_file("smiling_face.jpg")
landmarks = face_recognition.face_landmarks(image)

# Measure mouth distance for smile/frown detection
if landmarks:
    top_lip = landmarks[0]["top_lip"]
    bottom_lip = landmarks[0]["bottom_lip"]
    mouth_height = abs(top_lip[3][1] - bottom_lip[3][1])

    if mouth_height > 10:
        print("Smile detected!")
    else:
        print("Frown detected!")
```

---

### 24. **Face Recognition Attendance System with Time Logging**

```python
import cv2
import face_recognition
import csv
import datetime

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load known faces and encodings
known_faces = ["alice.jpg", "bob.jpg"]
known_names = ["Alice", "Bob"]
known_encodings = [face_recognition.face_encodings(face_recognition.load_image_file(img))[0] for img in known_faces]

# Open a CSV file to save attendance
attendance_file = open("attendance.csv", "w", newline="")
writer = csv.writer(attendance_file)
writer.writerow(["Name", "Time"])

# Set of recognized people
recognized_people = set()

while True:
    ret, frame = cap.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]

        # Record the name and time if not already recognized
        if name != "Unknown" and name not in recognized_people:
            recognized_people.add(name)
            writer.writerow([name, datetime.datetime.now()])
            print(f"Attendance marked for: {name}")

        # Draw bounding box and name
        for (top, right, bottom, left), name in zip(face_locations, [name] * len(face_locations)):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show frame with face recognition
    cv2.imshow("Face Recognition Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

attendance_file.close()
cap.release()
cv2.destroyAllWindows()
```

---

### 25. **Face Mask Detection with OpenCV (Using Pre-trained Models)**

```python
import cv2
import face_recognition

# Load pre-trained mask detector model (you can use any suitable model here)
mask_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the image
img = cv2.imread("masked_face.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = mask_cascade.detectMultiScale(gray, 1.1, 4)
for (x, y, w, h) in faces:
    face_region = img[y:y+h, x:x+w]
    # Implement mask/no-mask detection logic here using deep learning or heuristic rules
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow('Face Mask Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### 26. **Using Facial Recognition with Flask for Web Demo**

```python
from flask import Flask, render_template, request
import face_recognition
import cv2
import numpy as np

app = Flask(__name__)

# Load known faces and their encodings
known_images = ["alice.jpg", "bob.jpg"]
known_names = ["Alice", "Bob"]
known_encodings = [face_recognition.face_encodings(face_recognition.load_image_file(i))[0]
                   for i in known_images]

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files['file']
        img_path = "uploaded_image.jpg"
        file.save(img_path)

        image = face_recognition.load_image_file(img_path)
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        recognized_faces = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = known_names[first_match_index]
            recognized_faces.append(name)

        return render_template("result.html", recognized_faces=recognized_faces)

if __name__ == "__main__":
    app.run(debug=True)
```

### HTML Template: `index.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Upload Image for Face Recognition</title>
</head>
<body>
    <h2>Upload an Image</h2>
    <form action="/upload" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*">
        <input type="submit" value="Upload">
    </form>
</body>
</html>
```

---

### 27. **Emotion Detection using Facial Landmarks for Multiple Faces**

```python
import face_recognition
import cv2

# Load image with multiple faces
image = face_recognition.load_image_file("group_face.jpg")
face_locations = face_recognition.face_locations(image)
landmarks = face_recognition.face_landmarks(image)

# Process each face and detect emotion based on mouth and eye landmarks
for i, face_landmarks in enumerate(landmarks):
    # Example: Checking mouth width for detecting smile or frown
    top_lip = face_landmarks['top_lip']
    bottom_lip = face_landmarks['bottom_lip']
    mouth_width = abs(top_lip[0][0] - bottom_lip[6][0])

    # Define smile based on mouth width
    if mouth_width > 50:
        print(f"Person {i+1} is smiling.")
    else:
        print(f"Person {i+1} is not smiling.")
```

---

### 28. **Integrating Face Recognition with Social Media for Auto-Tagging**

```python
import face_recognition
import cv2
import requests
import json

# Example image from a social media post
image = face_recognition.load_image_file("social_media_image.jpg")
face_locations = face_recognition.face_locations(image)
face_encodings = face_recognition.face_encodings(image, face_locations)

# Assuming we have a function to fetch user details from a social media API
def fetch_social_media_user_data():
    # Simulated API response
    return {
        "users": [
            {"name": "Alice", "face_encoding": "alice_encoding_data"},
            {"name": "Bob", "face_encoding": "bob_encoding_data"}
        ]
    }

# Get known user data from API
user_data = fetch_social_media_user_data()
known_encodings = [user["face_encoding"] for user in user_data["users"]]
known_names = [user["name"] for user in user_data["users"]]

# Compare face encodings with the known users' encodings
for i, face_encoding in enumerate(face_encodings):
    matches = face_recognition.compare_faces(known_encodings, face_encoding)
    name = "Unknown"
    if True in matches:
        name = known_names[matches.index(True)]
    
    # Simulate tagging the recognized user in the image on social media
    if name != "Unknown":
        print(f"Tagging {name} in the post.")
    else:
        print(f"No match found for person {i+1}.")
```

---

### 29. **Deploying Face Recognition on Raspberry Pi (Edge Deployment)**

```python
import cv2
import face_recognition

# Set up camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces([known_face_encoding], face_encoding)
        name = "Unknown"
        if True in matches:
            name = "Known Face"

        # Draw rectangle around face and label it
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Face Recognition on Raspberry Pi", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Notes on Raspberry Pi:

* For deploying on Raspberry Pi, ensure you have the correct dependencies installed for OpenCV and face recognition. Use `pip install opencv-python face_recognition` to install them.
* Use the Raspberry Pi’s camera module for capturing video input.

---

### 30. **Cloud-Based Face Recognition with AWS Rekognition**

```python
import boto3
from PIL import Image
import io

# Initialize the AWS Rekognition client
rekognition_client = boto3.client('rekognition')

# Load an image to send to AWS Rekognition
with open("input_image.jpg", "rb") as img_file:
    img_bytes = img_file.read()

# Call AWS Rekognition for face detection
response = rekognition_client.detect_faces(
    Image={'Bytes': img_bytes},
    Attributes=['ALL']
)

# Print out the detected face details
for face in response['FaceDetails']:
    print("Detected face with:")
    print(f"Emotion: {face['Emotions'][0]['Type']}")
    print(f"Bounding Box: {face['BoundingBox']}")
```

### Notes on AWS Rekognition:

* You’ll need to set up AWS credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`) to use the Rekognition service.
* The `detect_faces` API provides detailed results, including emotions and bounding box information for the faces detected in the image.

---

### 31. **Using Facial Recognition with TensorFlow for Emotion Classification**

```python
import face_recognition
import cv2
import tensorflow as tf
import numpy as np

# Load pre-trained emotion classification model (e.g., EmotionNet)
emotion_model = tf.keras.models.load_model('emotion_model.h5')

# Load image and detect faces
image = face_recognition.load_image_file('input_face.jpg')
face_locations = face_recognition.face_locations(image)
face_encodings = face_recognition.face_encodings(image, face_locations)

# Classify emotion for each face
for face_encoding in face_encodings:
    # Pre-process face for emotion detection model (resize, normalize, etc.)
    face_image = face_recognition.face_landmarks(image)
    face_image = cv2.resize(face_image, (48, 48))
    face_image = np.expand_dims(face_image, axis=0)
    face_image = face_image / 255.0

    # Predict emotion
    emotion = emotion_model.predict(face_image)
    print(f"Predicted Emotion: {emotion}")
```

---

### 32. **Face Recognition for Virtual Reality Applications**

```python
import face_recognition
import cv2
import numpy as np
import openvr  # Virtual reality API

# Initialize VR system
vr_system = openvr.init(openvr.VRApplication_Scene)

# Load the image and detect faces
image = face_recognition.load_image_file("user_face.jpg")
face_locations = face_recognition.face_locations(image)

# Map detected face to VR coordinates for a virtual avatar
for (top, right, bottom, left) in face_locations:
    face_image = image[top:bottom, left:right]
    vr_coordinates = np.array([left, top, right, bottom])

    # Update virtual avatar based on the detected face
    vr_system.update_avatar(face_image, vr_coordinates)
    print("Updated virtual avatar with new face coordinates.")

# Close VR system
openvr.shutdown()
```

---

### 33. **Using Facial Recognition for Personal Security Systems**

```python
import cv2
import face_recognition

# Load known face encodings for the security system
known_face_encodings = [known_face_encoding]  # Example: pre-recorded face encodings
known_face_names = ["John Doe"]  # Example: pre-recorded names

# Initialize camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        
        if True in matches:
            name = known_face_names[matches.index(True)]

        if name != "Unknown":
            print(f"Access granted to {name}")
        else:
            print("Access denied!")

        # Draw rectangle and name on frame
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Personal Security System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

### 34. **Face Recognition for Real-Time Photo Tagging**

```python
import face_recognition
import cv2

# Load image and known faces
image = face_recognition.load_image_file("event_group_photo.jpg")
known_faces = ["alice.jpg", "bob.jpg"]
known_names = ["Alice", "Bob"]
known_encodings = [face_recognition.face_encodings(face_recognition.load_image_file(i))[0]
                   for i in known_faces]

# Detect faces in uploaded image
face_locations = face_recognition.face_locations(image)
face_encodings = face_recognition.face_encodings(image, face_locations)

for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces(known_encodings, face_encoding)
    name = "Unknown"
    
    if True in matches:
        name = known_names[matches.index(True)]

    print(f"Tagging {name} in photo.")

    # Optionally: Write tagged names back to image (e.g., for saving)
```

---

### 35. **Face Recognition in Augmented Reality (AR) Applications**

```python
import cv2
import face_recognition
import numpy as np
import cv2.aruco as aruco  # Augmented Reality library

# Initialize camera
cap = cv2.VideoCapture(0)

# Detect faces and overlay AR elements
while True:
    ret, frame = cap.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left) in face_locations:
        # Draw AR object (e.g., square) on top of face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Example: Draw an AR object (like an icon) around the face
        cv2.putText(frame, "AR Object", (left, top-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("AR Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

