import os
import lightgbm as lgb
import numpy as np

from face_detection import FaceDetector
detector = FaceDetector()

def load_data(folder_path):

        image_paths = []
        face_shape_categories = ["Heart", "Oval", "Oblong", "Round", "Square"]
        labels = []
        
        for category_index, category in enumerate(face_shape_categories):
                category_folder = os.path.join(folder_path, category)
            
                for image_name in os.listdir(category_folder):
                    image_path = os.path.join(category_folder, image_name)
                    image_paths.append(image_path)
                    labels.append(category_index)

        return image_paths, labels
current_dir = os.getcwd()
dataset_folder_training = os.path.join(current_dir, 'Dataset2\\training_set')
dataset_folder_testing = os.path.join(current_dir, 'Dataset2\\testing_set')

train_image_paths, train_labels = load_data(dataset_folder_training)

test_image_paths, test_labels = load_data(dataset_folder_testing)
    
    
train_features = [detector.get_measuremet_values(image_path) for image_path in train_image_paths]   
test_features = [detector.get_measuremet_values(image_path) for image_path in test_image_paths]

model = lgb.LGBMClassifier()
model.fit(train_features, train_labels)

importance = model.feature_importances_
importance_ratio = importance / np.sum(importance)
sorted_indices = np.argsort(importance)[::-1]

K = 20
selected_indices = sorted_indices[:K]

print("Selected Features:")
print(selected_indices)

print("Importance Ratios:")
print(importance_ratio[selected_indices])