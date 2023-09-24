import pickle
from face_detection import FaceDetector
import os
import cv2
import numpy as np

current_dir = os.getcwd()
img_path = os.path.join(current_dir, "Faces\\img5.jpg")
eyebrow_image_fpath = os.path.join(current_dir, "EyebrowShapes")
measurementDetect = FaceDetector()
import face_recognition

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
    new_measurement = measurementDetect.get_measuremet_values(new_image)
    combined_features = np.concatenate((face_encoding ,new_measurement), axis = 0)
    return combined_features

# Load the model from the file
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Assuming you have a new image named 'new_image.jpg'
new_image = img_path

# Extract measurements from the new image
new_measurement = extract_face_features(new_image)

# Make a prediction using the loaded model
prediction = model.predict([new_measurement])[0]

face_shape_categories = ["Heart", "Oval", "Oblong", "Round", "Square"]

face_shape_to_eyebrow = {
    0: 'Rounded shape brow',
    1: 'Soft angled brow',
    2: 'Straight brow',
    3: 'Hard-angled brow',
    4: 'S-shape brow'
}

eyebrow_suggestion = face_shape_to_eyebrow[prediction]

print(
    f"Predicted face shape: {face_shape_categories[prediction]}, Recommonded eyebrow shaping way: {eyebrow_suggestion}")

suggested_eyebrow_shape_path = os.path.join(eyebrow_image_fpath, f"{eyebrow_suggestion}.jpg")
suggested_eyebrow_shape = cv2.imread(suggested_eyebrow_shape_path)
image = cv2.imread(img_path)

suggested_eyebrow_shape = cv2.resize(suggested_eyebrow_shape, (300, suggested_eyebrow_shape.shape[0]))

result = np.copy(image)
result = cv2.resize(result,(300,300))
output = np.vstack((result,suggested_eyebrow_shape))

# Display the result
cv2.imshow('Result', output)
cv2.waitKey(0)
cv2.destroyAllWindows()