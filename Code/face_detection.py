import cv2
import dlib
import numpy as np
import math


class FaceDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            'haarcascade_frontalface_default.xml')
        self.predictor = dlib.shape_predictor(
            "shape_predictor_81_face_landmarks.dat")

    def detect_faces(self, image):
        faces = self.face_cascade.detectMultiScale(image, 1.1, 4)

        return faces

    def detect_landmarks(self, image, faces):
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            dlib_rect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
            detected_landmarks = self.predictor(image, dlib_rect).parts()
            landmarks_matrix = np.matrix(
                [[l.x, l.y] for l in detected_landmarks])

            landmark_image = image.copy()
            for idx, point in enumerate(landmarks_matrix):
                pos = (point[0, 0], point[0, 1])
                cv2.putText(landmark_image, str(
                    idx), pos, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 0, 255))
            cv2.circle(landmark_image, pos, 3, color=(0, 255, 255))

        return landmarks_matrix

    def calculate_distance(self, landmarks, point1, point2):
        x1 = landmarks[int(point1), 0]
        y1 = landmarks[int(point1), 1]
        x2 = landmarks[int(point2), 0]
        y2 = landmarks[int(point2), 1]

        x_diff = (x1 - x2) ** 2
        y_diff = (y1 - y2) ** 2

        distance = math.sqrt(x_diff + y_diff)
        return distance

    def get_face_height(self, image, landmarks):
        start_point = (landmarks[8, 0], landmarks[8, 1])
        end_point = (landmarks[27, 0], landmarks[27, 1])

        theta = np.arctan2(start_point[1]-end_point[1],
                           start_point[0]-end_point[0])
        end_x = int(start_point[0] - 10000 * np.cos(theta))
        end_y = int(start_point[1] - 10000 * np.sin(theta))

        extended_point = (end_x, end_y)

        line_point1 = (landmarks[69, 0], landmarks[69, 1])
        line_point2 = (landmarks[72, 0], landmarks[72, 1])

        intersect = self.line_intersect(line_point1[0], line_point1[1], line_point2[0], line_point2[1],
                                        start_point[0], start_point[1], extended_point[0], extended_point[1])

        cv2.circle(image, (int(intersect[0]), int(
            intersect[1])), 5, color=(255, 255, 255), thickness=-1)

        p2 = (int(intersect[0]), int(intersect[1]))

        xd = (int(start_point[0]) - p2[0]) ** 2
        yd = (int(start_point[1]) - p2[1]) ** 2

        distance = math.sqrt(xd + yd)
        return distance

    def calculate_jaw_angles(self, landmarks):
        point1 = (landmarks[1, 0], landmarks[1, 1])
        point2 = (landmarks[4, 0], landmarks[4, 1])  
        point3 = (landmarks[6, 0], landmarks[6, 1])  
        point4 = (landmarks[8, 0], landmarks[8, 1])  
        end_point1 = (landmarks[28, 0], landmarks[28, 1])
        end_point2 = (landmarks[48, 0], landmarks[48, 0])
        end_point3 = (landmarks[10, 0], landmarks[10, 0])

        angle1 = self.calculate_angle(end_point1, point2, point1)
        angle2 = self.calculate_angle(end_point2, point3, point2)
        angle3 = self.calculate_angle(end_point3, point4, point3)

        return angle1, angle2, angle3

    def calculate_jawline_angles(self, point1, point2, point3):
        vector1 = (point1[0] - point2[0], point1[1] - point2[1])
        vector2 = (point3[0] - point2[0], point3[1] - point2[1])

        angle_radiants = math.atan2(vector1,vector2)

    def calculate_angle(self, point1, point2, point3):
        vector1 = (point1[0] - point2[0], point1[1] - point2[1])
        vector2 = (point3[0] - point2[0], point3[1] - point2[1])

        dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
        magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
        magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

        cosine_angle = dot_product / (magnitude1 * magnitude2)
        angle = math.degrees(math.acos(cosine_angle))

        return angle

    def line_intersect(self, Ax1, Ay1, Ax2, Ay2, Bx1, By1, Bx2, By2):
        """ returns a (x, y) tuple or None if there is no intersection """
        d = (By2 - By1) * (Ax2 - Ax1) - (Bx2 - Bx1) * (Ay2 - Ay1)
        if d:
            uA = ((Bx2 - Bx1) * (Ay1 - By1) - (By2 - By1) * (Ax1 - Bx1)) / d
            uB = ((Ax2 - Ax1) * (Ay1 - By1) - (Ay2 - Ay1) * (Ax1 - Bx1)) / d
        else:
            return
        if not (0 <= uA <= 1 and 0 <= uB <= 1):
            return
        x = Ax1 + uA * (Ax2 - Ax1)
        y = Ay1 + uA * (Ay2 - Ay1)

        return x, y

    def get_face_measurements(self, image, face_coordinates):

        face_height = self.get_face_height(image, face_coordinates)

        jawline_length = self.calculate_distance(face_coordinates, 8, 3)

        forehead_width = self.calculate_distance(face_coordinates, 75, 79)

        cheekbone_width = self.calculate_distance(face_coordinates, 26, 17)

        chin_width = self.calculate_distance(face_coordinates, 10, 6)

        jaw_width = self.calculate_distance(face_coordinates, 4, 12)

        face_width = self.calculate_distance(face_coordinates, 1, 15)

        lawer_chin_width = self.calculate_distance(face_coordinates, 7, 9)

        chin_to_mouth_width = self.calculate_distance(face_coordinates, 8, 57)

        normalize_distance = self.calculate_distance(face_coordinates, 0, 16)

        return face_height, jawline_length, forehead_width, cheekbone_width, chin_width, jaw_width, \
               face_width, lawer_chin_width,chin_to_mouth_width, normalize_distance

    def get_measuremet_values(self, filepath):
        image = cv2.imread(filepath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detect_faces(gray)
        features = []
        angles = []

        for face in faces:
            face_coordinates = self.detect_landmarks(image, faces)
            measurements = self.get_face_measurements(image, face_coordinates)
            angles = self.calculate_jaw_angles(face_coordinates)
            p1 = (face_coordinates[57, 0], face_coordinates[57, 1])
            p2 = (face_coordinates[8, 0], face_coordinates[8, 1])
            p3 = (face_coordinates[7, 0], face_coordinates[7, 1])
            chin_angle = self.calculate_angle(p1, p2, p3)

            forehead_width = measurements[2]
            cheekbone_width = measurements[3]
            face_length = measurements[0]
            jawline_length = measurements[1]
            chin_width = measurements[4]
            jaw_width = measurements[5]
            face_width = measurements[6]
            lower_chin_width = measurements[7]
            chin_to_mouth_width = measurements[8]
            normalize_distance = measurements[9]

            # Normalize Values
            n_forehead_width = forehead_width / normalize_distance
            n_cheekbone_width = cheekbone_width / normalize_distance
            n_face_length = face_length / normalize_distance
            n_jawline_length = jawline_length / normalize_distance
            n_chin_width = chin_width / normalize_distance
            n_face_width = face_width / normalize_distance
            n_jaw_width = jaw_width / normalize_distance
            n_lower_chin_width = lower_chin_width / normalize_distance
            n_chin_to_mouth_width =chin_to_mouth_width /normalize_distance

            ratio1 = n_face_width / n_face_length
            ratio2 = n_forehead_width / n_face_length
            ratio3 = n_cheekbone_width / n_face_width
            ratio4 = n_jaw_width / n_face_width
            ratio5 = n_forehead_width/n_face_width
            ratio6 = n_chin_width/n_face_width
            ratio7 = n_jaw_width/n_forehead_width
            ratio8 = n_lower_chin_width / n_chin_width
            ratio9 = n_chin_to_mouth_width / n_jaw_width

            features.extend([n_forehead_width, n_cheekbone_width, n_face_length, n_jawline_length, n_chin_width,
                            n_face_width, n_jaw_width, n_chin_to_mouth_width,
                            angles[0], angles[1], angles[2],
                            ratio1, ratio9])

            return features