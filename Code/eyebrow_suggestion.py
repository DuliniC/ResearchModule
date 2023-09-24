import pickle
from face_detection import FaceDetector
import os
import cv2
import numpy as np

current_dir = os.getcwd()
eyebrow_image_fpath = os.path.join(current_dir, "EyebrowShapes")
face_shape_categories = ["Heart", "Oval", "Oblong", "Round", "Square"]
face_shape_to_eyebrow = {
    0: 'Rounded shape brow',
    1: 'Soft angled brow',
    2: 'Straight brow',
    3: 'Hard-angled brow',
    4: 'S-shape brow'
}

measurementDetect = FaceDetector()
import face_recognition

class EyebrowShapingWaySuggestion:
    def extract_face_features(self, image_path):
        img = cv2.imread(image_path)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_img)

        if len(face_locations) == 0:
            return None
            
        face_encoding = face_recognition.face_encodings(rgb_img, known_face_locations=face_locations)[0]
        new_measurement = measurementDetect.get_measuremet_values(image_path)
        combined_features = np.concatenate((face_encoding ,new_measurement), axis = 0)
        return combined_features

    def shape_prediction(self,img_path):
        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)

        new_image = img_path

        new_measurement = self.extract_face_features(new_image)

        prediction = model.predict([new_measurement])[0]

        return prediction


    def suggestion(self, prediction,img_path):
        eyebrow_suggestion = face_shape_to_eyebrow[prediction]
        output_msg =(f"Predicted face shape: {face_shape_categories[prediction]}, \
                     Recommonded eyebrow shaping way: {eyebrow_suggestion}")
        print(output_msg)

        suggested_eyebrow_shape_path = os.path.join(eyebrow_image_fpath, f"{eyebrow_suggestion}.jpg")
        suggested_eyebrow_shape = cv2.imread(suggested_eyebrow_shape_path)
        image = cv2.imread(img_path)

        suggested_eyebrow_shape = cv2.resize(suggested_eyebrow_shape, (300, suggested_eyebrow_shape.shape[0]))

        result = np.copy(image)
        result = cv2.resize(result,(300,300))
        output = np.vstack((result,suggested_eyebrow_shape))

        return output, output_msg
        # Display the result
        # cv2.imshow('Result', output)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def main(self, img_path):
        prediction = self.shape_prediction(img_path)
        suggestion_img, output_msg = self.suggestion(prediction, img_path)
        return suggestion_img, output_msg

        # cv2.imshow('Result', suggestion_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()   
             
ob = EyebrowShapingWaySuggestion()