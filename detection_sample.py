# -*- coding: utf-8 -*-
# @Author: yirui
# @Date:   2021-11-15 17:42:59
# @Last Modified by:   yirui
# @Last Modified time: 2021-11-15 20:11:09
from mtcnn import MTCNN
import cv2
import numpy as np
from mtcnn_alignment import FaceAligner

detector = MTCNN(weights_file='models/detection/mtcnn_weights.npy', min_face_size=20)
image = cv2.cvtColor(cv2.imread("test_images/target.jpg"), cv2.COLOR_BGR2RGB)
results = detector.detect_faces(image)

fa = FaceAligner()

cropped = []
for res in results:
    bbox = res['box']
    landmarks = [res['keypoints']['left_eye'],res['keypoints']['right_eye']]
    aligned = fa.align(image, landmarks)
    cropped.append(aligned)
# cv2.imshow('immage', image)
cv2.imshow('aligned', cropped[0])
cv2.waitKey(0)
