import cv2
import numpy as np
import gc


import FaceFeatureDetection, ModelFactory, ImageFeatureDraw, FeatureProcessing

face_model = ModelFactory.get_face_detector(modelFile="models/res10_300x300_ssd_iter_140000.caffemodel",
                                            configFile="models/deploy.prototxt")

landmark_model = ModelFactory.get_landmark_model('models/pose_model')

AVG_COUNT = 5

class(API):

    def __init__(self):
        # Variable init.
        self.face_pos_avg = {'x': 0, 'y': 0, 'x1': 0, 'y1': 0}
        self.face_ang_avg = {'x': 0, 'y': 0}




    def sum_face_position(self, faces, face_pos_sum):
        for x, y, x1, y1 in faces:
            # I got better results without averaging the landmarks
            # But we can play around more
            face_pos_sum['x'] = x
            face_pos_sum['y'] = y
            face_pos_sum['x1'] = x1
            face_pos_sum['y1'] = y1

    def run(self):
        face_pos_sum =

        face_ang_sum['x'] += x_angle
        face_ang_sum['y'] += y_angle
        if count % AVG_COUNT == 0:
            # This seems redundant if we are not averaging, but I will leave
            # for experimental purposes
            x_loc_avg = face_pos_sum['x']
            y_loc_avg = face_pos_sum['y']
            x1_loc_avg = face_pos_sum['x1']
            y1_loc_avg = face_pos_sum['y1']
            x_ang_avg = face_ang_sum['x']/AVG_COUNT
            y_ang_avg = face_ang_sum['y']/AVG_COUNT

            x_ang_cond = x_ang_avg >= 5 or x_ang_avg <= -7
            y_ang_cond = y_ang_avg >= 5 or y_ang_avg <= -5
            x_pos_cond = not (size[1] / 4 < x_loc_avg <
                              size[1] / 2 < x1_loc_avg < 3 * size[1] / 4)
            y_pos_cond = False  # checking for y seems useless for laptop cameras

            flags = 0
            if x_pos_cond or y_pos_cond:
                print("Wombo works best with your face horizontally centered")
                flags += 1
            if x_ang_cond or y_ang_cond:
                print("Please look straight at the camera")
                flags += 1
            if flags == 0:
                print("Congrats for being a decent human being")

            # reset averages/sums
            face_ang_sum = {'x': 0, 'y': 0}
            face_pos_sum = {'x': 0, 'y': 0, 'x1': 0, 'y1': 0}
