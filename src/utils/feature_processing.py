import cv2
import numpy as np
import math




def get_nose_endpoint(rotation_vector, translation_vector, camera_matrix, dist_coeffs):
    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                     translation_vector, camera_matrix, dist_coeffs)
    return nose_end_point2D


def open_mouth_detector(face_landmark_points) -> bool:
    """ Returns true if the mouth detected in the image is open (or smiling with teeth)
    :param image: np.ndarray
    :return: bool
    Example usage:
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rects = find_faces(img, face_model)
    if len(rects) > 1:
        print("Multiple faces detected...")
    shape = detect_marks(img, landmark_model, rects[0])
    open_mouth_detector(shape)
    Output: bool
    """

    outer_points = [[49, 59], [50, 58], [51, 57], [52, 56], [53, 55]]
    d_outer = [0] * 5
    inner_points = [[61, 67], [62, 66], [63, 65]]
    d_inner = [0] * 3

    cnt_outer = 0
    cnt_inner = 0
    for i, (p1, p2) in enumerate(outer_points):
        try:
            if d_outer[i] + 3 < face_landmark_points[p2][1] - face_landmark_points[p1][1]:
                cnt_outer += 1
        except:
            pass
    for i, (p1, p2) in enumerate(inner_points):
        try:
            if d_inner[i] + 2 < face_landmark_points[p2][1] - face_landmark_points[p1][1]:
                cnt_inner += 1
        except:
            pass
    if cnt_outer > 2.5 and cnt_inner > 2:
        return True
    else:
        return False
