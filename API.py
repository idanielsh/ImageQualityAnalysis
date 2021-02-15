import cv2
import numpy as np
import FaceFeatureDetection, ModelFactory, ImageFeatureDraw, FeatureProcessing


# Face model to find faces in an image
face_model = ModelFactory.get_face_detector(modelFile="models/res10_300x300_ssd_iter_140000.caffemodel",
                                            configFile="models/deploy.prototxt")

# Landmark model for figuring out where facial features are
landmark_model = ModelFactory.get_landmark_model('models/pose_model')

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
    return FaceFeatureDetection.find_faces(img, face_model)


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
    return FaceFeatureDetection.detect_marks(img, face, landmark_model)

def get_node_end_point(img, marks, camera_matrix):
    """
    Calculates an (x,y) tuple of where the user is looking on the screen.

    :param img: The image The image in which the marks are to be found
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


    return FaceFeatureDetection.head_pose_points(img, rotation_vector, translation_vector, camera_matrix)


def get_glance_estimate(img, marks, camera_matrix):
    """

    :return:
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

    (x,y) = FaceFeatureDetection.head_pose_points(img, rotation_vector, translation_vector, camera_matrix)

    FeatureProcessing.get_look_angles(marks[0], rotation_vector, translation_vector,
                                      camera_matrix, np.zeros((4, 1)), x, y)


