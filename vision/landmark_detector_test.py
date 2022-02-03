from unittest.case import TestCase
from core.face_landmarks import format_landmarks
from vision.landmark_detector import Detector
from vision.utils.image_loader import image_from_path
from imageio import imread
import cv2
import numpy

class LandmarkDetectorTest(TestCase):

    def test_pytorch(self):
        self._test('./models/detection/landmarks.pth')

    def test_onnx(self):
        self._test('./models/detection/landmarks.onnx')

    def _test(self, model_path):
        image = imread(image_from_path('./test_images/test6.png'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detector = Detector(model_path)
        landmarks = detector.detect(image, numpy.array([0, 0, 473, 460], dtype=numpy.int32))
        self.assertDictEqual(
            format_landmarks(landmarks, ndigits=1),
            {
                'bounds': {
                    'topLeft': [0.0, 0.0],
                    'bottomRight': [473.0, 460.0]
                },
                'landmarks': {
                    'outline': [[93.1, 188.8], [95.1, 220.8], [101.6, 250.1], [111.8, 278.4], [125.3, 304.2], [145.5, 327.6], [167.1, 344.1], [192.9, 356.4], [228.3, 359.5], [264.0, 348.0], [294.1, 329.5], [319.7, 307.4], [340.3, 280.4], [350.6, 251.6], [356.6, 221.0], [358.6, 189.7], [356.7, 157.5]],
                    'left_brows': [[106.9, 152.9], [118.2, 138.9], [136.5, 132.1], [156.4, 131.9], [175.1, 136.1]],
                    'right_brows': [[232.4, 128.0], [251.4, 119.3], [273.2, 114.9], [296.2, 117.2], [315.6, 127.8]],
                    'nose_back': [[206.3, 164.4], [206.0, 184.5], [205.7, 204.3], [206.7, 223.2]],
                    'nostrils': [[190.2, 242.6], [200.3, 244.4], [212.8, 245.4], [225.8, 241.3], [237.9, 236.9]],
                    'left_eye': [[131.7, 176.2], [142.3, 166.9], [159.3, 165.1], [176.1, 172.5], [160.7, 178.1], [144.0, 180.5]],
                    'right_eye': [[245.3, 163.8], [258.7, 152.4], [275.6, 150.2], [291.5, 156.6], [277.6, 163.9], [260.0, 165.9]],
                    'mouth': [[175.1, 285.4], [187.9, 274.9], [204.4, 267.3], [215.3, 268.0], [226.0, 264.2], [246.8, 266.9], [266.9, 273.4], [248.8, 288.6], [232.4, 296.7], [218.6, 299.9], [205.2, 300.5], [191.1, 295.8], [181.8, 284.7], [205.5, 278.9], [217.2, 277.4], [229.7, 275.3], [260.0, 274.0], [230.2, 281.6], [217.7, 284.4], [205.7, 285.0]]
                },
                'angles': {
                    'pitch': 6.122, 'yaw': 5.526, 'roll': -5.594
                }
            }
        )
