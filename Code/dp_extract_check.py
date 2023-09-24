import cv2
import numpy as np
import face_recognition
import os
from face_detection import FaceDetector
detector = FaceDetector()

# Function to extract face features
def extract_face_features(image_path):
    face_encoding_a = []
    # Load and preprocess the image
    img = cv2.imread(image_path)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect faces in the image
    face_locations = face_recognition.face_locations(rgb_img)

    if len(face_locations) == 0:
        return None
    # Extract face features from the first detected face
    face_encoding = face_recognition.face_encodings(rgb_img, known_face_locations=face_locations)[0]
    a = np.array(face_encoding).tolist()
    return face_encoding

# Example usage
current_dir = os.getcwd()
img_path = os.path.join(current_dir, "Faces\\img4.jpg")
face_features = extract_face_features(img_path)
face_features2 = detector.get_measuremet_values(img_path)
combined_features = np.concatenate((face_features2,face_features), axis =0)

if face_features is not None:
    print(combined_features)
else:
    print("No face found in the image.")
