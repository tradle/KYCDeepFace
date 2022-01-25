import json
from unittest.case import TestCase
from core.face_landmarks import FaceLandmarks, format_landmarks
import numpy as np

test_nostrlis = [[121.53, 59.63], [121.46, 60.08], [121.63, 60.68], [123.13, 61.16], [125.33, 61.57]]
test_left_brows = [[129.09, 47.26], [133.51, 46.87], [136.38, 47.2], [138.07, 48.24], [138.25, 49.57]]
test_right_brows = [[140.58, 52.07], [143.93, 52.37], [147.18, 53.12], [149.75, 54.58], [150.86, 56.12]]
test_outline = [
    [ 96.28, 50.95], [ 89.8,  54.52], [ 85.05, 57.98], [ 79.84, 60.56], [ 84.09, 64.0 ],
    [ 90.49, 66.56], [ 97.29, 68.48], [104.79, 70.37], [113.55, 72.59], [125.57, 72.56],
    [139.6,  71.69], [153.57, 70.56], [165.5,  68.71], [176.5,  65.6 ], [177.64, 63.29],
    [181.34, 60.31], [183.19, 57.26]
]
test_nose_back = [[134.8, 54.11], [130.96, 55.51], [127.22, 56.79], [123.81, 58.36]]
test_left_eye = [[128.6, 51.21], [131.17, 51.42], [132.89, 52.29], [132.38, 53.61], [130.16, 53.06], [128.56, 52.21]]
test_right_eye = [[136.58, 56.3], [140.06, 56.46], [142.26, 57.15], [143.28, 58.07], [139.72, 57.99], [137.3, 57.26]]
test_mouth = [
    [114.33, 63.23], [115.83, 62.31], [118.49, 62.1],
    [119.67, 62.71], [121.57, 63.13], [124.11, 64.75], [126.53, 67.05], [119.81, 67.07],
    [117.53, 67.1], [115.1, 66.59], [113.22, 65.87], [113.06, 64.36], [115.11, 63.39],
    [117.65, 63.61], [119.0, 64.05], [120.91, 64.69], [124.4, 66.77], [119.23, 65.83],
    [117.19, 65.3], [115.6, 64.7]
]
test_landmarks = np.concatenate([
    test_outline,
    test_left_brows,
    test_right_brows,
    test_nose_back,
    test_nostrlis,
    test_left_eye,
    test_right_eye,
    test_mouth
])

def deepEquals(test: TestCase, a, b, msg, fn):
    test.assertEqual(len(a), len(b), f'len({a}) {msg}')
    for index, entry in enumerate(a):
        fn(entry, b[index], f'[{index}] {msg}')

def deepTupleEquals(test: TestCase, a, b, msg=''):
    deepEquals(test, a, b, msg, lambda a, b, msg: test.assertTupleEqual(a, b, msg))

def deepListEquals(test: TestCase, a, b, msg=''):
    deepEquals(test, a, b, msg, lambda a, b, msg: test.assertListEqual(list(a), list(b), msg))

class FaceLandmarkTest(TestCase):
    def test_sample(self):
        landmarks = FaceLandmarks(test_landmarks, (300, 300), np.array([0., 10., 20., 30.]))
        deepListEquals(self, list(landmarks.outline), test_outline, 'outline')
        deepListEquals(self, list(landmarks.left_brows), test_left_brows, 'left_brows')
        deepListEquals(self, list(landmarks.right_brows), test_right_brows, 'right_brows')
        deepListEquals(self, list(landmarks.nostrils), test_nostrlis, 'nostrlis')
        deepListEquals(self, list(landmarks.nose_back), test_nose_back, 'nose_back')
        deepListEquals(self, list(landmarks.left_eye), test_left_eye, 'left_eye')
        deepListEquals(self, list(landmarks.right_eye), test_right_eye, 'right_eye')
        deepListEquals(self, list(landmarks.mouth), test_mouth, 'mouth')
        self.assertListEqual(list(landmarks.outline.first_pt), test_outline[0])
        self.assertListEqual(list(landmarks.outline.last_pt), test_outline[-1])
        self.assertListEqual(list(landmarks.outline.middle_pt), test_outline[8])
        self.assertListEqual(list(landmarks.left_eye.center), [130.62666666666667, 52.29999999999999])
        # Ubuntu and mac have slightly differing number point operations?!
        self.assertAlmostEqual(landmarks.angle[0], 67.82354739260106)
        self.assertAlmostEqual(landmarks.angle[1], -59.09734920451225)
        self.assertAlmostEqual(landmarks.angle[2], 3.890346326419562)
        self.assertListEqual(list(landmarks.eye_line.center), [135.24666666666667, 54.75249999999999])
        deepListEquals(self, list(landmarks.eye_line), [landmarks.left_eye.center, landmarks.right_eye.center], 'eye_line')
        self.assertEqual(json.dumps(format_landmarks(landmarks), indent=4), '''{
    "bounds": {
        "topLeft": [
            0.0,
            10.0
        ],
        "bottomRight": [
            20.0,
            30.0
        ]
    },
    "landmarks": {
        "outline": [
            [
                96.28,
                50.95
            ],
            [
                89.8,
                54.52
            ],
            [
                85.05,
                57.98
            ],
            [
                79.84,
                60.56
            ],
            [
                84.09,
                64.0
            ],
            [
                90.49,
                66.56
            ],
            [
                97.29,
                68.48
            ],
            [
                104.79,
                70.37
            ],
            [
                113.55,
                72.59
            ],
            [
                125.57,
                72.56
            ],
            [
                139.6,
                71.69
            ],
            [
                153.57,
                70.56
            ],
            [
                165.5,
                68.71
            ],
            [
                176.5,
                65.6
            ],
            [
                177.64,
                63.29
            ],
            [
                181.34,
                60.31
            ],
            [
                183.19,
                57.26
            ]
        ],
        "left_brows": [
            [
                129.09,
                47.26
            ],
            [
                133.51,
                46.87
            ],
            [
                136.38,
                47.2
            ],
            [
                138.07,
                48.24
            ],
            [
                138.25,
                49.57
            ]
        ],
        "right_brows": [
            [
                140.58,
                52.07
            ],
            [
                143.93,
                52.37
            ],
            [
                147.18,
                53.12
            ],
            [
                149.75,
                54.58
            ],
            [
                150.86,
                56.12
            ]
        ],
        "nose_back": [
            [
                134.8,
                54.11
            ],
            [
                130.96,
                55.51
            ],
            [
                127.22,
                56.79
            ],
            [
                123.81,
                58.36
            ]
        ],
        "nostrils": [
            [
                121.53,
                59.63
            ],
            [
                121.46,
                60.08
            ],
            [
                121.63,
                60.68
            ],
            [
                123.13,
                61.16
            ],
            [
                125.33,
                61.57
            ]
        ],
        "left_eye": [
            [
                128.6,
                51.21
            ],
            [
                131.17,
                51.42
            ],
            [
                132.89,
                52.29
            ],
            [
                132.38,
                53.61
            ],
            [
                130.16,
                53.06
            ],
            [
                128.56,
                52.21
            ]
        ],
        "right_eye": [
            [
                136.58,
                56.3
            ],
            [
                140.06,
                56.46
            ],
            [
                142.26,
                57.15
            ],
            [
                143.28,
                58.07
            ],
            [
                139.72,
                57.99
            ],
            [
                137.3,
                57.26
            ]
        ],
        "mouth": [
            [
                114.33,
                63.23
            ],
            [
                115.83,
                62.31
            ],
            [
                118.49,
                62.1
            ],
            [
                119.67,
                62.71
            ],
            [
                121.57,
                63.13
            ],
            [
                124.11,
                64.75
            ],
            [
                126.53,
                67.05
            ],
            [
                119.81,
                67.07
            ],
            [
                117.53,
                67.1
            ],
            [
                115.1,
                66.59
            ],
            [
                113.22,
                65.87
            ],
            [
                113.06,
                64.36
            ],
            [
                115.11,
                63.39
            ],
            [
                117.65,
                63.61
            ],
            [
                119.0,
                64.05
            ],
            [
                120.91,
                64.69
            ],
            [
                124.4,
                66.77
            ],
            [
                119.23,
                65.83
            ],
            [
                117.19,
                65.3
            ],
            [
                115.6,
                64.7
            ]
        ]
    },
    "angles": {
        "pitch": 67.824,
        "yaw": -59.097,
        "roll": 3.89
    }
}''')