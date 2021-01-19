# This class will launch your camera and show the detection algorithm
import cv2
import numpy as np

import FaceFeatureDetection, ModelFactory, ImageFeatureDraw

face_model = ModelFactory.get_face_detector(modelFile="models/res10_300x300_ssd_iter_140000.caffemodel",
                                            configFile="models/deploy.prototxt")

landmark_model = FaceFeatureDetection.get_landmark_model()

# Loads the video capture
cap = cv2.VideoCapture(0)
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

# Iterating through the image capture. Stop script by pressing q
while True:
    ret, img = cap.read()
    if ret == True:
        # Faces is a list of tuples which are the bounding box of the detected face
        faces = FaceFeatureDetection.find_faces(img, face_model)
        for face in faces:
            # Draws a bounding box around each face
            ImageFeatureDraw.draw_faces(img, faces)

            # Returns an array of all the features. I'm not sure what all of them are other than the ones stored in image_points
            marks = FaceFeatureDetection.detect_marks(img, landmark_model, face)

            image_points = np.array([
                marks[30],  # Nose tip
                marks[8],  # Chin
                marks[36],  # Left eye left corner
                marks[45],  # Right eye right corne
                marks[48],  # Left Mouth corner
                marks[54]  # Right mouth corner
            ], dtype="double")

            # Not sure what this does
            dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                          dist_coeffs, flags=cv2.SOLVEPNP_UPNP)
