import cv2
import numpy as np
from src.utils import face_feature_detection, model_factory, feature_processing

# Face model to find faces in an image
face_model = model_factory.get_face_detector(modelFile="src/models/res10_300x300_ssd_iter_140000.caffemodel",
                                             configFile="src/models/deploy.prototxt")

# Landmark model for figuring out where facial features are
landmark_model = model_factory.get_landmark_model('src/models/pose_model')

# 3D model locations of facial features relative to the nose
model_points = np.array([
    (0.0, 0.0, 0.0),  # Nose tip
    (0.0, -330.0, -65.0),  # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),  # Right eye right corner
    (-150.0, -150.0, -125.0),  # Left Mouth corner
    (150.0, -150.0, -125.0)  # Right mouth corner
])


def find_faces_faces(img):
    """
    Find the faces in an image

    Parameters
    ----------
    img : np.uint8
        Image to find faces from
    Returns
    -------
    faces : list
        List of coordinates of the faces detected in the image
    :return:
    """
    return face_feature_detection.find_faces(img, face_model)


def detect_marks(img, face):
    """
    Find the facial landmarks in an image from the faces

    Parameters
    ----------
    img : np.uint8
        The image in which landmarks are to be found
    face : list
        Face coordinates (x, y, x1, y1) in which the landmarks are to be found

    Returns
    -------
    marks : numpy array
        facial landmark points

    """
    return face_feature_detection.detect_marks(img, face, landmark_model)


def get_nose_end_point(img, marks, camera_matrix):
    """
    Calculates an (x,y) tuple of where the user is looking on the screen.

    :param img: The image in which the marks are to be found
    :param marks: facial landmark points
    :param camera_matrix: The camera matrix
    :return: (x, y) : tuple
        Coordinates of line to estimate head pose of where the user is looking
    """
    image_points = np.array([
        marks[30],  # Nose tip
        marks[8],  # Chin
        marks[36],  # Left eye left corner
        marks[45],  # Right eye right corner
        marks[48],  # Left Mouth corner
        marks[54]  # Right mouth corner
    ], dtype="double")

    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, np.zeros((4, 1)))

    return face_feature_detection.head_pose_points(img, rotation_vector, translation_vector, camera_matrix)


def get_glance_angle_estimate(img, marks, camera_matrix):
    """
    Returns the angle that the user is looking in degress
    :param img: The image in which the marks are to be found
    :param marks: facial landmark points
    :param camera_matrix: The camera matrix
    :return: an (x_angle, y_angle) tuple of the angle that the user is looking.
    """
    image_points = np.array([
        marks[30],  # Nose tip
        marks[8],  # Chin
        marks[36],  # Left eye left corner
        marks[45],  # Right eye right corner
        marks[48],  # Left Mouth corner
        marks[54]  # Right mouth corner
    ], dtype="double")

    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, np.zeros((4, 1)))

    (x, y) = face_feature_detection.head_pose_points(img, rotation_vector, translation_vector, camera_matrix)

    return feature_processing.get_look_angles(marks[0], rotation_vector, translation_vector,
                                              camera_matrix, np.zeros((4, 1)), x, y)


def open_mouth_detector(marks):
    """
    Returns true if the mouth detected in the image is open (or smiling with teeth)
    :param image: np.ndarray
    :return: bool
    """

    return feature_processing.open_mouth_detector(marks)


def x_location(img, face) -> float:
    """
    Returns the relative x-location of the face to the center of the image.
    -1 implies the face is on the very left side of the screen
    0 implies the face is perfectly centered horizontally
    1 implies the face is on the very right side of the screen

    :param face: The face being assessed, a list of pixels (x0, y0, x1, y1)
    :param img: The image being assessed
    :return: a float as the relative position of the face
    """
    midpoint = (face[0] + face[2]) / 2
    width = img.shape[1]

    if midpoint < 0:
        return -1
    if midpoint > width:
        return 1

    return midpoint / (width / 2) - 1


def y_location(img, face) -> float:
    """
    Returns the relative y-location of the face to the center of the image.
    -1 implies the face is on the very bottom of the screen
    0 implies the face is perfectly centered vertically
    1 implies the face is on the very top side of the screen

    :rtype: float
    :param face: The face being assessed, a list of pixels (x0, y0, x1, y1)
    :param img: The image being assessed
    :return: a float as the relative position of the face
    """
    midpoint = (face[1] + face[3]) / 2
    height = img.shape[0]

    if midpoint < 0:
        return -1
    if midpoint > height:
        return 1

    return 1 - midpoint / (height / 2)
