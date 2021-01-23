import cv2
import numpy as np
import csv

import FaceFeatureDetection, ModelFactory, FeatureProcessing

face_model = ModelFactory.get_face_detector(modelFile="models/res10_300x300_ssd_iter_140000.caffemodel",
                                            configFile="models/deploy.prototxt")

video_dir = 'stats/videos'
video_name = 'test.mp4'

csv_dir = 'stats/video_analysis'
csv_name = video_name + '.csv'

cap = cv2.VideoCapture(f'{video_dir}/{video_name}')

ret, img = cap.read()
size = img.shape
font = cv2.FONT_ITALIC

# 3D model locations of facial features relative to the nose
model_points = np.array([
    (0.0, 0.0, 0.0),  # Nose tip
    (0.0, -330.0, -65.0),  # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),  # Right eye right corner
    (-150.0, -150.0, -125.0),  # Left Mouth corner
    (150.0, -150.0, -125.0)  # Right mouth corner
])

# Camera setup
focal_length = size[1]
center = (size[1] / 2, size[0] / 2)
camera_matrix = np.array(
    [[focal_length, 0, center[0]],
     [0, focal_length, center[1]],
     [0, 0, 1]], dtype="double"
)
frame = 0
with open(f'{csv_dir}/{csv_name}', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["frame", "x_angle", "y_angle"])
    while cap.isOpened():
        ret, img = cap.read()
        if ret == True:
            # Faces is a list of tuples which are the bounding box of the detected face
            faces = FaceFeatureDetection.find_faces(img, face_model)
            # Draws a bounding box around each face
            for face in faces:
                # Returns an array of all the features. I'm not sure what all of them are other than the ones stored in image_points
                marks = FaceFeatureDetection.detect_marks(img, face, 'models/pose_model')

                image_points = np.array([
                    marks[30],  # Nose tip
                    marks[8],  # Chin
                    marks[36],  # Left eye left corner
                    marks[45],  # Right eye right corne
                    marks[48],  # Left Mouth corner
                    marks[54]  # Right mouth corner
                ], dtype="double")

                # SolvePNP calculates where the person is looking
                dist_coeffs = np.zeros((4, 1))
                (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                              dist_coeffs)  # , flags=cv2.SOLVEPNP_UPNP)

                x1, x2 = FaceFeatureDetection.head_pose_points(img, rotation_vector, translation_vector, camera_matrix)

                # Calculates the angle of where the person is looking using the nose end point for the line.
                (x_angle, y_angle) = FeatureProcessing.get_look_angles(image_points[0], rotation_vector, translation_vector,
                                                                       camera_matrix, dist_coeffs, x1, x2)

                print(frame)
                writer.writerow([frame, x_angle, y_angle])

            # cv2.imshow('img', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
        frame += 1
cv2.destroyAllWindows()
cap.release()

