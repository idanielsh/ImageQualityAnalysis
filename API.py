import cv2
import numpy as np
import gc


import FaceFeatureDetection, ModelFactory, ImageFeatureDraw, FeatureProcessing

face_model = ModelFactory.get_face_detector(modelFile="models/res10_300x300_ssd_iter_140000.caffemodel",
                                            configFile="models/deploy.prototxt")

landmark_model = ModelFactory.get_landmark_model('models/pose_model')