from unittest.case import TestCase
from core.face_landmarks import format_landmarks
from vision.landmark_detector import Detector
from vision.utils.image_loader import image_from_path
from imageio import imread
import cv2
import numpy

class LandmarkDetectorTest(TestCase):
    def test_sample(self):
        image = imread(image_from_path('./test_images/test6.png'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detector = Detector(model_path='./models/detection/landmarks.pth')
        landmarks = detector.detect(image, numpy.array([0, 0, 473, 460], dtype=numpy.int32))
        self.assertDictEqual(
            format_landmarks(landmarks),
            {
                'bounds': {
                    'topLeft': [0.0, 0.0],
                    'bottomRight': [473.0, 460.0]
                },
                'landmarks': {
                    'outline': [
                        [93.05, 188.78], [95.14, 220.77], [101.57, 250.14], [111.82, 278.37], [125.31, 304.24], [145.45, 327.64], [167.12, 344.14], [192.9, 356.36], [228.27, 359.47], [264.04, 347.97], [294.07, 329.51], [319.66, 307.39], [340.29, 280.39], [350.63, 251.64], [356.61, 220.98], [358.55, 189.75], [356.66, 157.48]
                    ],
                    'left_brows': [[106.89, 152.91], [118.25, 138.9], [136.53, 132.09], [156.37, 131.89], [175.08, 136.09]],
                    'right_brows': [[232.36, 128.02], [251.44, 119.26], [273.23, 114.93], [296.2, 117.21], [315.64, 127.8]],
                    'nose_back': [[206.29, 164.38], [205.97, 184.49], [205.67, 204.27], [206.74, 223.24]],
                    'nostrils': [[190.22, 242.61], [200.32, 244.42], [212.76, 245.42], [225.82, 241.28], [237.94, 236.88]],
                    'left_eye': [[131.72, 176.15], [142.26, 166.94], [159.34, 165.1], [176.05, 172.49], [160.71, 178.09], [143.95, 180.51]],
                    'right_eye': [[245.28, 163.8], [258.71, 152.45], [275.64, 150.22], [291.53, 156.62], [277.64, 163.9], [259.97, 165.93]],
                    'mouth': [[175.09, 285.38], [187.93, 274.85], [204.42, 267.26], [215.26, 268.0], [225.97, 264.21], [246.83, 266.91], [266.92, 273.36], [248.79, 288.57], [232.43, 296.72], [218.56, 299.87], [205.17, 300.55], [191.1, 295.79], [181.78, 284.65], [205.46, 278.89], [217.17, 277.43], [229.75, 275.28], [259.97, 273.96], [230.21, 281.57], [217.69, 284.38], [205.66, 285.02]]
                },
                'angles': {
                    'pitch': 6.122, 'yaw': 5.526, 'roll': -5.594
                }
            }
        )
