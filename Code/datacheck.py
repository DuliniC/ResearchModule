from PIL import Image
import os
from face_detection import FaceDetector
detector = FaceDetector()

current_dir = os.getcwd()
folder_path = os.path.join(current_dir, "published_dataset")
# Specify the path to the folder containing the images

# Specify the desired size for resizing
target_size = (299, 299)
face_shape_categories = ["heart", "oval", "oblong", "round", "square"]

for category_index, category in enumerate(face_shape_categories):
    category_folder = os.path.join(folder_path, category)
            
    # Iterate through all files in the folder
    for filename in os.listdir(category_folder):
        image_path = os.path.join(category_folder, filename)
        # Check if the file is an image
        if os.path.isfile(image_path) or filename.lower().endswith(('.jpg', '.jpeg')):

            # Open the image file
            try:
            #  img = Image.open(image_path)

                # Attempt to resize the image
                #resized_img = img.resize(target_size)

                # Check if the resized image has the desired size
                #if resized_img.size != target_size:
                #    print(f"Image file '{filename}' couldn't be resized to {target_size}.") """
                detector = FaceDetector()
                measure_face_coordinates = detector.get_measuremet_values(image_path)
                if not measure_face_coordinates or measure_face_coordinates is None:
                    print(f"{category},{filename}") 

            except (IOError, SyntaxError) as e:
                # Handle any errors that occur during image processing
                print(f"Error processing image file '{filename}': {e}")

""" import cv2
import os


folder_path = 'Dataset/training_set/Oblong/'

# Iterate over the files in the folder
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    
    # Load the image
    image = cv2.imread(file_path)
    
    if image is not None:
        landmark_image = image.copy()
        # Perform other operations on the image
    else:
        print(f"Failed to load the image: {file_path}")
 """
