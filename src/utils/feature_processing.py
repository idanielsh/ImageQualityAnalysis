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
    Output: True
    """

    outer_points = [[49, 59], [50, 58], [51, 57], [52, 56], [53, 55]]
    d_outer = [0] * 5
    inner_points = [[61, 67], [62, 66], [63, 65]]
    d_inner = [0] * 3


    cnt_outer = 0
    cnt_inner = 0
    for i, (p1, p2) in enumerate(outer_points):
        if d_outer[i] + 3 < face_landmark_points[p2][1] - face_landmark_points[p1][1]:
            cnt_outer += 1
    for i, (p1, p2) in enumerate(inner_points):
        if d_inner[i] + 2 < face_landmark_points[p2][1] - face_landmark_points[p1][1]:
            cnt_inner += 1
    if cnt_outer > 2.5 and cnt_inner > 2:
        return True
    else:
        return False