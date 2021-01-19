import cv2
import numpy as np
import math

def get_look_angles(nose, rotation_vector, translation_vector, camera_matrix, dist_coeffs, headpose_x1, headpose_x2):
    nose_end_point2D = get_nose_endpoint(rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    # Position of the nose
    p1 = (int(nose[0]), int(nose[1]))

    # Position of nose endpoint
    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

    try:
        m = (p2[1] - p1[1]) / (p2[0] - p1[0])
        ang1 = int(math.degrees(math.atan(m)))
    except:
        ang1 = 90

    try:
        m = (headpose_x2[1] - headpose_x1[1]) / (headpose_x2[0] - headpose_x1[0])
        ang2 = int(math.degrees(math.atan(-1 / m)))
    except:
        ang2 = 90

    return (ang2, ang1)



def get_nose_endpoint(rotation_vector, translation_vector, camera_matrix, dist_coeffs):
    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                     translation_vector, camera_matrix, dist_coeffs)
    return nose_end_point2D