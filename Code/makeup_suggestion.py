import cv2
import numpy as np
import os
import math

class MakeoverSuggestion : 

    def estimate_skin_tone(self, image):
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_image)
        ita1 = ((l_channel - 50) / b_channel) * (180 / np.pi)
        ita = np.mean(ita1)
        print(ita)
        if np.logical_and(0 >= ita, ita < 11):
            return "dark"
        elif np.logical_and(11 <= ita, ita < 42):
            return "medium"
        else:
            return "fair"

    def estimate_skin_tone2(self, image):
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

        if len(faces) == 0:
            return None 

        x, y, w, h = faces[0]
        face_image = image[y:y + h, x:x + w]
        face_image = image
        lab_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2LAB)

        lower_skin_tone = np.array([0, 20, 77], dtype=np.uint8)
        upper_skin_tone = np.array([255, 200, 130], dtype=np.uint8)
        skin_mask = cv2.inRange(lab_image, lower_skin_tone, upper_skin_tone)
        num_skin_pixels = cv2.countNonZero(skin_mask)

        total_pixels = image.shape[0] * image.shape[1]
        skin_tone_percentage = (num_skin_pixels / total_pixels) * 100
        print(skin_tone_percentage)
        if skin_tone_percentage >= 40:
            return 'fair'
        elif skin_tone_percentage >= 10:
            return 'medium'
        else:
            return 'dark'

    def suggest_makeup_skin_tone(self, skin_tone):
        if skin_tone == 'fair':
            return ['Light pink lipstick shades', 'Soft brown colour eyeshadow', 'Peach colour blush']
        elif skin_tone == 'medium':
            return ['Rose lipstick shades', 'Bronze colour eyeshadow', 'Warm coral colour blush']
        elif skin_tone == 'dark':
            return ['Berry lipstick shades', 'Gold colour eyeshadow', 'Rich plum colour blush']
        else:
            return []

    def main(self, image_path):
        image = cv2.imread(image_path)
        skin_tone = obj.estimate_skin_tone(image)
        #print("Estimated skin tone:", skin_tone)

        suggested_makeup = obj.suggest_makeup_skin_tone(skin_tone)
        #print("Suggested Makeup for", skin_tone, "skin tone:", suggested_makeup)

        return skin_tone, suggested_makeup

obj = MakeoverSuggestion()
current_dir = os.getcwd()
image_path = os.path.join(current_dir, 'Faces\\heart.jpg')
obj.main(image_path)

# image = cv2.imread(image_path)
# skin_tone = obj.estimate_skin_tone(image)
# print("Estimated skin tone:", skin_tone)

# suggested_makeup = obj.suggest_makeup_skin_tone(skin_tone)
# print("Suggested Makeup for", skin_tone, "skin tone:", suggested_makeup)
