from face_detection import FaceDetector
import math
detector = FaceDetector()

class ShapePredictor:


    def classify_face_shape(self, face_measurements):
        forehead_width = face_measurements['forehead_width']
        cheekbone_width = face_measurements['cheekbone_width']
        face_length = face_measurements['face_length']
        jawline_length = face_measurements['jawline_length']
        chin_width = face_measurements['chin_width']
        face_width = face_measurements['face_width']
        jaw_width = face_measurements['jaw_width']
        angle1 = face_measurements['angle1']
        angle2 = face_measurements['angle2']
        angle3 = face_measurements['angle3']
        chin_angle = face_measurements['chin_angle']

        n_forehead_width = forehead_width / jaw_width
        n_cheekbone_width = cheekbone_width / jaw_width
        n_face_length = face_length / jaw_width
        n_jawline_length = jawline_length / jaw_width
        n_chin_width = chin_width / jaw_width
        n_face_width = face_width / jaw_width


        # if (forehead_width > cheekbone_width and forehead_width > jaw_width) and \
        #         (angle1 > angle2 and angle3 > angle2):
        #     return 'Heart'
        # elif (face_length > cheekbone_width and face_length > forehead_width and face_length > jaw_width) and \
        #         (angle1 > 90 and angle2 > angle3):
        #     return 'Oblong'
        # elif (face_length > cheekbone_width and forehead_width > jaw_width) and \
        #         math.isclose(angle1, angle2, abs_tol=5) and math.isclose(angle2,angle3,abs_tol=5):
        #     return 'Oval'
        # elif (math.isclose(face_length,cheekbone_width,abs_tol=45) and math.isclose(cheekbone_width,forehead_width,abs_tol=25) and \
        #         math.isclose(forehead_width,jaw_width,abs_tol=25)) and (angle1 > 90 and math.isclose(angle2,angle3,abs_tol=5)):
        #     return 'Square'
        # elif (math.isclose(face_length,cheekbone_width,abs_tol=45) and math.isclose(forehead_width,jaw_width,abs_tol=45)) and \
        #         (math.isclose(angle1,angle3,abs_tol=20) and angle2 > angle3):
        #     return 'Round'
        # else:
        #     return 'Oval'
        if abs(angle1 - angle2) >=10 and abs(angle2-angle3)<=40 and \
             (face_length > cheekbone_width and face_length > forehead_width):
            category = "Oblong"
        elif angle1 > 50 and angle2 > 40 :
            category = "Heart"
        elif (forehead_width - jaw_width) > 0.31 and abs(angle1 - angle3) <= 50 and (face_length/cheekbone_width)<1.6:
            category = "Round"
        elif (face_length > cheekbone_width and abs(cheekbone_width - forehead_width) <= 2 and \
            abs(forehead_width - jawline_length) <= 2 and abs(angle1 - 90) <= 5 and abs(angle2 - angle3) <= 5):
            category = "Square"
        else:
            category = "Oval"

        return category

        # el
        # if abs(jawline_width - forehead_width) <= 30 :
        #     category = "Oval"
        # elif abs(angle1 - angle3) <= 50:
        #     category = "Square"
        # else:
        #     category = "Oval2"

        # if n_forehead_width > n_jawline_length and angle1 > angle2 > angle3:
        #     return "Heart"
        # elif face_length > cheekbone_width and abs(cheekbone_width - forehead_width) <= 2 and \
        #     abs(forehead_width - jawline_length) <= 2 and abs(angle1 - 90) <= 2 and angle2 > angle3 :
        #     return "Oblong"
        # elif face_length > cheekbone_width and forehead_width > jawline_length and \
        #      abs(angle1 - angle2) <= 2 and abs(angle2 - angle3) <= 2:
        #     return "Oval"
        # elif face_length > cheekbone_width and abs(cheekbone_width - forehead_width) <= 2 and \
        #     abs(forehead_width - jawline_length) <= 2 and abs(angle1 - 90) <= 2 and abs(angle2 - angle3) <= 2:
        #     return "Square"
        # elif face_length > cheekbone_width and abs(face_length - cheekbone_width) <= 2 and abs(forehead_width - jawline_length) <= 2:
        #     return "Round"
        # else:
        #     return "Oval"