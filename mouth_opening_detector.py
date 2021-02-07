# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 01:04:44 2020
@author: hp
"""

import cv2
from face_detector import get_face_detector, find_faces
from face_landmarks import get_landmark_model, detect_marks, draw_marks



def run_live():
    face_model = get_face_detector()
    landmark_model = get_landmark_model()
    outer_points = [[49, 59], [50, 58], [51, 57], [52, 56], [53, 55]]
    d_outer = [0] * 5
    inner_points = [[61, 67], [62, 66], [63, 65]]
    d_inner = [0] * 3
    font = cv2.FONT_HERSHEY_SIMPLEX
    cap = cv2.VideoCapture(0)
    while (True):
        ret, img = cap.read()
        rects = find_faces(img, face_model)
        for rect in rects:
            shape = detect_marks(img, landmark_model, rect)
            draw_marks(img, shape)
            cv2.putText(img, 'Press r to record Mouth distances', (30, 30), font,
                        1, (0, 255, 255), 2)
            cv2.imshow("Output", img)
        if cv2.waitKey(1) & 0xFF == ord('r'):
            for i in range(100):
                for i, (p1, p2) in enumerate(outer_points):
                    d_outer[i] += shape[p2][1] - shape[p1][1]
                for i, (p1, p2) in enumerate(inner_points):
                    d_inner[i] += shape[p2][1] - shape[p1][1]
            break
    cv2.destroyAllWindows()
    d_outer[:] = [x / 100 for x in d_outer]
    d_inner[:] = [x / 100 for x in d_inner]

    while (True):
        ret, img = cap.read()
        rects = find_faces(img, face_model)
        for rect in rects:
            shape = detect_marks(img, landmark_model, rect)
            cnt_outer = 0
            cnt_inner = 0
            draw_marks(img, shape[48:])
            for i, (p1, p2) in enumerate(outer_points):
                if d_outer[i] + 3 < shape[p2][1] - shape[p1][1]:
                    cnt_outer += 1
            for i, (p1, p2) in enumerate(inner_points):
                if d_inner[i] + 2 < shape[p2][1] - shape[p1][1]:
                    cnt_inner += 1
            if cnt_outer > 2.5 and cnt_inner > 2:
                print('Wombo works best if you save that smile for after :)')
                cv2.putText(img, 'Smiling', (30, 30), font,
                            1, (0, 255, 255), 2)
            # show the output image with the face detections + facial landmarks
        cv2.imshow("Output", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def open_mouth_detector(image) -> bool:
    """ Returns true if the mouth detected in the image is open (or smiling with teeth)

    :param image: np.ndarray
    :return: bool

    Example usage:
    my_image = cv2.imread("person_with_mouth_open.jpg")
    open_mouth_detector(my_image)

    Output: True
    """
    face_model = get_face_detector()
    landmark_model = get_landmark_model()
    outer_points = [[49, 59], [50, 58], [51, 57], [52, 56], [53, 55]]
    d_outer = [0] * 5
    inner_points = [[61, 67], [62, 66], [63, 65]]
    d_inner = [0] * 3

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    rects = find_faces(img, face_model)
    if len(rects) > 1:
        print("Multiple faces detected...")

    shape = detect_marks(img, landmark_model, rects[0])

    cnt_outer = 0
    cnt_inner = 0
    draw_marks(img, shape[48:])
    for i, (p1, p2) in enumerate(outer_points):
        if d_outer[i] + 3 < shape[p2][1] - shape[p1][1]:
            cnt_outer += 1
    for i, (p1, p2) in enumerate(inner_points):
        if d_inner[i] + 2 < shape[p2][1] - shape[p1][1]:
            cnt_inner += 1
    if cnt_outer > 2.5 and cnt_inner > 2:
        print('Wombo works best if you save that smile for after :)')
        return True
    else:
        return False




my_img = cv2.imread("utsav-womboai.jpg")
print(open_mouth_detector(my_img))
