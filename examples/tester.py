import cv2
import numpy as np

# api for accessing program
from src import image_analysis_services

# Methods for drawing features on the image
import image_feature_draw

# Loads the video capture from default camera
# Use cap = cv2.VideoCapture('path/to/video.mp4') to load a file
cap = cv2.VideoCapture(0)

ret, img = cap.read()


# Camera setup
center = (img.shape[1] / 2, img.shape[0] / 2)
camera_matrix = np.array(
    [[img.shape[1], 0, center[0]],
     [0, img.shape[1], center[1]],
     [0, 0, 1]], dtype="double"
)

while True:
    # Captures a frame
    ret, img = cap.read()
    if ret == True:

        # A list of the faces detected in the image
        faces = image_analysis_services.find_faces_faces(img)
        for face in faces:
            # Detects 68 points on the face
            marks = image_analysis_services.detect_marks(img, face)

            image_feature_draw.draw_all_marks(img, marks)

            # Returns the relative horizontal position of the center of the face
            x_position = image_analysis_services.x_location(img, face)

            # Returns the relative vertical position of the center of the face
            y_postion = image_analysis_services.y_location(img, face)

            # Find if the mouth of the face is open
            mouth_open = image_analysis_services.open_mouth_detector(marks);

            (x,y) = image_analysis_services.get_nose_end_point(img, marks, camera_matrix)

            (x_deg, y_deg) = image_analysis_services.get_glance_angle_estimate(img, marks, camera_matrix);

    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()