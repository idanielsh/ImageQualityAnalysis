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
print('Press \'q\' to exit!')

frame = 0

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

            # Finds the endpoint of where the nose is looking
            (x, y) = image_analysis_services.get_nose_end_point(img, marks, camera_matrix)

            # Finds the degrees of where the person is looking
            (x_deg, y_deg) = image_analysis_services.get_glance_angle_estimate(img, marks, camera_matrix);

            # Note that this algorithm only prints every 25 frames for clarity

            dy = 1.2*cv2.getTextSize('aaaaaa', cv2.QT_FONT_NORMAL, 0.75, 1)[0][1]

            cv2.putText(img, f'Face: {face}', (int((x_position + 1) * center[0]), int((1 - y_postion) * center[1])),
                        cv2.QT_FONT_NORMAL, 0.75, (255, 255, 255))
            cv2.putText(img, f' Face centered at {(round(x_position, 1), round(y_postion, 1))}',
                        (int((x_position + 1) * center[0]), int((1 - y_postion) * center[1] + dy) ), cv2.QT_FONT_NORMAL,
                        0.75, (255, 255, 255))
            cv2.putText(img, f' Face glancing at angle: {(round(x_deg), round(y_deg))}',
                        (int((x_position + 1) * center[0]), int((1 - y_postion) * center[1] + + 2 * dy) ),
                        cv2.QT_FONT_NORMAL, 0.75, (255, 255, 255))
            cv2.putText(img, ' Mouth: ' + ('open' if mouth_open else 'closed'),
                        (int((x_position + 1) * center[0]), int((1 - y_postion) * center[1] + + 3 * dy)),
                        cv2.QT_FONT_NORMAL, 0.75, (255, 255, 255))

            # print()
            # print(f'    Face: {face}:')
            # print(f'    Face centered at {(x_position, y_postion)} relative to image center')
            # print(f'    Face glancing at angle: {(x_deg, y_deg)}')

    cv2.imshow('ImageQualityAnalysis', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame += 1

cv2.destroyAllWindows()
cap.release()
