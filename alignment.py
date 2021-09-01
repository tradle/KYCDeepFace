# -*- coding: utf-8 -*-
# @Author: yirui
# @Date:   2021-09-01 17:39:28
# @Last Modified by:   yirui
# @Last Modified time: 2021-09-01 17:40:40
import cv2
import numpy as np

import argparse
import os
import sys
import json
import time
import torch
import imageio
import scipy.io
import torch.utils.data
from vision.ssd.config.fd_config import define_img_size
from core import model as mfn
from core.utils import *
from landmark_detector import Detector as landmark_detector
from mask_predictor import Detector as mask_detector
from config import *
import pickle

class FaceAligner:
    def __init__(self, desiredLeftEye=(0.35, 0.35),
        desiredFaceWidth=256, desiredFaceHeight=None):
        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight
        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth

    def align(self, image, landmarks):
        rightEyePts = landmarks[36:42]
        leftEyePts = landmarks[42:48]
        # compute the center of mass for each eye
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")
        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180
        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]
        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
            (leftEyeCenter[1] + rightEyeCenter[1]) // 2)
        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
        # update the translation component of the matrix
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        # apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, h),
            flags=cv2.INTER_CUBIC)
        # return the aligned face
        return output

if __name__ == '__main__':
    device = torch.device("cpu")
    define_img_size(DETECTION_INPUT_SIZE)
    from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
    from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
    class_names = [name.strip() for name in open(DETECTION_LABEL).readlines()]
    num_classes = len(class_names)
    model_path = DETECTION_FAST_MODEL_PATH
    det_net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=TEST_DEVICE)
    det_predictor = create_Mb_Tiny_RFB_fd_predictor(det_net, candidate_size=DETECTION_CANDIDATE_SIZE, device=device)
    det_net.load(model_path)
    landmark_predictor = landmark_detector(test_device=device)

    fa = FaceAligner(desiredLeftEye=(0.30,0.30), desiredFaceWidth=112)

    cap = cv2.VideoCapture(0)
    while True:
        ret, orig_image = cap.read()

        if orig_image is None:
            print("end")
            break

        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        image_height, image_width, _ = image.shape

        boxes, labels, probs = det_predictor.predict(image, DETECTION_CANDIDATE_SIZE / 2, DETECTION_THRESHOLD)

        for i in range(boxes.size(0)):
            box = boxes[i, :]
            # cv2.rectangle(orig_image, pos_tuple((int(box[0]), int(box[1]))), pos_tuple((int(box[2]), int(box[3]))), (255, 255, 255), 2)

            landmark, angle = landmark_predictor.detect(orig_image, box.numpy())
            # print(angle)
            t0 = time.time()
            aligned = fa.align(orig_image, landmark)
            print(f"time used - {time.time() - t0}")
        cv2.imshow(f'aligned', aligned)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
