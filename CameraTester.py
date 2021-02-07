# This class will launch your camera and show the detection algorithm
import cv2
import numpy as np
import gc


import FaceFeatureDetection, ModelFactory, ImageFeatureDraw, FeatureProcessing

face_model = ModelFactory.get_face_detector(modelFile="models/res10_300x300_ssd_iter_140000.caffemodel",
                                            configFile="models/deploy.prototxt")

landmark_model = ModelFactory.get_landmark_model('models/pose_model')

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


count = 0
# Iterating through the image capture. Stop script by pressing q
while True:
    ret, img = cap.read()
    if ret == True:
        # Faces is a list of tuples which are the bounding box of the detected face
        faces = FaceFeatureDetection.find_faces(img, face_model)
        # Draws a bounding box around each face
        ImageFeatureDraw.draw_faces(img, faces)

        # Variable init.
        facial_ang_sum = {'x': 0, 'y': 0}
        facial_pos_sum = {'x': 0, 'y': 0, 'x1': 0, 'y1': 0}
        ITER = 5
        for face in faces:
            count += 1

            # Returns an array of all the features. I'm not sure what all of them are other than the ones stored in image_points
            marks = FaceFeatureDetection.detect_marks(img, face, landmark_model)

            # Draws all of the marks collected by the model on the image
            ImageFeatureDraw.draw_all_marks(img, marks)


            image_points = np.array([
                marks[30],  # Nose tip
                marks[8],   # Chin
                marks[36],  # Left eye left corner
                marks[45],  # Right eye right corner
                marks[48],  # Left Mouth corner
                marks[54]   # Right mouth corner
            ], dtype="double")
            print(marks[48], marks[54])
            # Not sure what solvePnP does
            dist_coeffs = np.zeros((4, 1))
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                   dist_coeffs)  # flags=cv2.SOLVEPNP_UPNP


            x1, x2 = FaceFeatureDetection.head_pose_points(img, rotation_vector, translation_vector, camera_matrix)

            # Calculates the nose end point for the line.
            (x_angle, y_angle) = FeatureProcessing.get_look_angles(image_points[0], rotation_vector, translation_vector,
                                                                   camera_matrix, dist_coeffs, x1, x2)
            for x, y, x1, y1 in faces:
                # I got better results without averaging the landmarks
                # But we can play around more
                facial_pos_sum['x'] = x
                facial_pos_sum['y'] = y
                facial_pos_sum['x1'] = x1
                facial_pos_sum['y1'] = y1

            facial_ang_sum['x'] += x_angle
            facial_ang_sum['y'] += y_angle
            if count % ITER == 0:
                # This seems redundant if we are not averaging, but I will leave
                # for experimental purposes
                x_loc_avg = facial_pos_sum['x']
                y_loc_avg = facial_pos_sum['y']
                x1_loc_avg = facial_pos_sum['x1']
                y1_loc_avg = facial_pos_sum['y1']
                x_ang_avg = facial_ang_sum['x']/ITER
                y_ang_avg = facial_ang_sum['y']/ITER

                x_ang_cond = x_ang_avg >= 5 or x_ang_avg <= -7
                y_ang_cond = y_ang_avg >= 5 or y_ang_avg <= -5
                x_pos_cond = not (size[1] / 4 < x_loc_avg <
                                  size[1] / 2 < x1_loc_avg < 3 * size[1] / 4)
                y_pos_cond = False  # checking for y seems useless for laptop cameras

                flags = 0
                total_flags = 3
                feedback = ""
                if x_pos_cond or y_pos_cond:
                    feedback += "center your face, "
                    flags += 1
                if x_ang_cond or y_ang_cond:
                    feedback += "look straight at the camera, "
                    flags += 1
                if flags == 0:
                    feedback = "keep being an awesome human being!"

                print(f'{count}. You have {flags}/{total_flags}. Please, {feedback}')

                # reset averages/sums
                facial_ang_sum = {'x': 0, 'y': 0}
                facial_pos_sum = {'x': 0, 'y': 0, 'x1': 0, 'y1': 0}

        cv2.imshow('img', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
    gc.collect()
cv2.destroyAllWindows()
cap.release()
