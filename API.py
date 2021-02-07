import cv2
import numpy as np
import gc

import FaceFeatureDetection, ModelFactory, ImageFeatureDraw, FeatureProcessing

face_model = ModelFactory.get_face_detector(modelFile="models/res10_300x300_ssd_iter_140000.caffemodel",
                                            configFile="models/deploy.prototxt")

landmark_model = ModelFactory.get_landmark_model('models/pose_model')


def get_marks(img, face):
    return FaceFeatureDetection.detect_marks(img, face, landmark_model)


def find_faces(img):
    return FaceFeatureDetection.find_faces(img, face_model)


def calculate_camera_angle(marks, ):
    image_points = np.array([
        marks[30],  # Nose tip
        marks[8],  # Chin
        marks[36],  # Left eye left corner
        marks[45],  # Right eye right corne
        marks[48],  # Left Mouth corner
        marks[54]  # Right mouth corner
    ], dtype="double")


    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                  dist_coeffs)  # flags=cv2.SOLVEPNP_UPNP

    x1, x2 = FaceFeatureDetection.head_pose_points(img, rotation_vector, translation_vector, camera_matrix)

    # Calculates the nose end point for the line.
    (x_angle, y_angle) = FeatureProcessing.get_look_angles(image_points[0], rotation_vector, translation_vector,
                                                           camera_matrix, dist_coeffs, x1, x2)