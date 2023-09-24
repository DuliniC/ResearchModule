import os
from sklearn.base import BaseEstimator, ClassifierMixin
from classify_face_shape import ShapePredictor
from face_detection import FaceDetector
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow import keras

shapePredict = ShapePredictor()
measurementDetect = FaceDetector()
current_dir = os.getcwd()
dataset_folder = os.path.join(current_dir, "Dataset2\\training_set")

class RuleBasedFaceShapeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def predict(self, X):
        predictions = []
        for measurement in X:
            prediction = shapePredict.classify_face_shape(measurement)
            predictions.append(prediction)
        return predictions

class modelTrain:
    def __init__(self):

        face_shape_categories = ['Heart', 'Oblong', 'Oval', 'Square', 'Round']

        self.images = []
        self.labels = []

        for category_index, category in enumerate(face_shape_categories):
            category_folder = os.path.join(dataset_folder, category)
            
            for image_name in os.listdir(category_folder):
                image_path = os.path.join(category_folder, image_name)
                self.images.append(image_path)
                
                self.labels.append(category_index)

    def train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.images, self.labels, test_size=0.2, random_state=42)

        self.measurements_train = []
        for image in X_train:
            measurement = measurementDetect.get_measuremet_values(image)
            if(measurement is not None):
                self.measurements_train.append(measurement)

        self.measurements_test = []
        for image in X_test:
            measurement = measurementDetect.get_measuremet_values(image)
            if(measurement is not None):
                self.measurements_test.append(measurement)

        classifier = RuleBasedFaceShapeClassifier()

        classifier.fit(self.measurements_train, y_train)

        y_pred = classifier.predict(self.measurements_test)

        return X_train, X_test, y_train, y_test,y_pred, classifier

shape_mapping = {
    'Heart': 0,
    'Oblong': 1,
    'Oval': 2,
    'Square': 3,
    'Round': 4
}

t_model = modelTrain()
X_train, X_test, y_train, y_test,y_pred, classifier = t_model.train_model()
y_pred_update = [shape_mapping[shape_name] for shape_name in y_pred]
accuracy = accuracy_score(y_test, y_pred_update)
print(f"Accuracy: {accuracy}")
import pickle

with open('trained_model.pkl', 'wb') as file:
    pickle.dump(classifier, file)
