import dlib
import numpy as np
import cv2
import os
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

from face_detection import FaceDetector
from classify_face_shape import ShapePredictor
import csv
import time
detector = FaceDetector()

class ModelSelector:
    # Function to run the model and calculate accuracy
    def run_model(self,X_train, y_train, X_test, y_test, model):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        return acc

    def main(self):
        current_dir = os.getcwd()
        dataset_folder = os.path.join(current_dir, "Dataset2\\training_set")
        features = []

        # Load your dataset and labels
        face_shape_categories = ["Heart", "Oval", "Oblong", "Round", "Square"]
        labels = []

        # Initialize variables
        accuracy = 0

        # Loop until desired accuracy is reached
        while accuracy <= 0.9:
            # Extract facial landmarks and features
            for category_index, category in enumerate(face_shape_categories):
                category_folder = os.path.join(dataset_folder, category)
            
                for image_name in os.listdir(category_folder):
                    image_path = os.path.join(category_folder, image_name)
                    extracted_features = detector.get_measuremet_values(image_path)
                    features.append(extracted_features)
                    labels.append(category_index)
            
            # Perform PCA for feature reduction
            pca = PCA(n_components=10)
            reduced_features = pca.fit_transform(features)
            
            # Split the dataset into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(reduced_features, labels, test_size=0.2, random_state=42)
            
            # List of classifiers to try
            classifiers = [SVC(),RandomForestClassifier(),LGBMClassifier()]
            
            # Train and evaluate each classifier
            for classifier in classifiers:
                classifier_name = classifier.__class__.__name__
                print(f"Training and evaluating {classifier_name}...")
                acc = self.run_model(X_train, y_train, X_test, y_test, classifier)
                print(f"Accuracy for {classifier_name}: {acc}")
                if acc > accuracy:
                    accuracy = acc
                    best_model = classifier.__class__.__name__
            
            # Return the features, best model, and accuracy
        return reduced_features, best_model, accuracy

ob = ModelSelector()

reduced_features, best_model, accuracy = ob.main()

print(f"Best model: {best_model}")
print(f"Best accuracy: {accuracy}")
print(reduced_features)