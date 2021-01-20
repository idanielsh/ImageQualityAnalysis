import cv2
from tensorflow import keras


def get_face_detector(modelFile=None,
                      configFile=None,
                      quantized=False):
    """
    Get the face detection caffe model of OpenCV's DNN module

    Parameters
    ----------
    modelFile : string, optional
        Path to model file. The default is "models/res10_300x300_ssd_iter_140000.caffemodel" or models/opencv_face_detector_uint8.pb" based on quantization.
    configFile : string, optional
        Path to config file. The default is "models/deploy.prototxt" or "models/opencv_face_detector.pbtxt" based on quantization.
    quantization: bool, optional
        Determines whether to use quantized tf model or unquantized caffe model. The default is False.

    Returns
    -------
    model : dnn_Net
    :param quantized:
    """
    if quantized:
        if modelFile is None:
            modelFile = "models/opencv_face_detector_uint8.pb"
        if configFile is None:
            configFile = "models/opencv_face_detector.pbtxt"
        model = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

    else:
        if modelFile is None:
            modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
        if configFile is None:
            configFile = "models/deploy.prototxt"
        model = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return model

def get_landmark_model(saved_model='models/pose_model'):
    """
    Get the facial landmark model.
    Original repository: https://github.com/yinguobing/cnn-facial-landmark

    Parameters
    ----------
    saved_model : string, optional
        Path to facial landmarks model. The default is 'models/pose_model'.

    Returns
    -------
    model : Tensorflow model
        Facial landmarks model

    """
    model = keras.models.load_model(saved_model)
    return model
