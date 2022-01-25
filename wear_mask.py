import os
import sys
import argparse
import numpy as np
import cv2
import random

import math
from PIL import Image, ImageFile

import json
import time
import torch
import imageio
import scipy.io
import torch.utils.data
from core import model as mfn
from core.utils import *
from config import *

from alignment import FaceAligner
from vision.landmark_detector import Detector as landmark_detector

import time

__version__ = '0.3.0'


IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'masks')
DEFAULT_IMAGE_PATH = os.path.join(IMAGE_DIR, 'default-mask.png')
BLACK_IMAGE_PATH = os.path.join(IMAGE_DIR, 'black-mask.png')
BLUE_IMAGE_PATH = os.path.join(IMAGE_DIR, 'blue-mask.png')
RED_IMAGE_PATH = os.path.join(IMAGE_DIR, 'red-mask.png')

masks = ["masks/mask1.png","masks/mask2.png","masks/mask3.png","masks/mask4.png","masks/mask5.png","masks/mask6.png"]

def face_capture(img, file=False):
    if file:
        orig_img = cv2.imread(img)
    else:
        orig_img = img
    image = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    print(img)
    boxes, labels, probs = det_predictor.predict(image, DETECTION_CANDIDATE_SIZE / 2, DETECTION_THRESHOLD)
    # assume that when registering, only one face would be detected.
    box = boxes[0, :]
    x1, y1, x2, y2 = pos_box(box)
    print(x1, y1, x2, y2)
    aligned_face = image[y1:y2, x1:x2]
    landmark, _ = landmark_predictor.detect(orig_image, box.numpy())
    # temp_check = './temp'
    # cv2.imwrite(os.path.join(temp_check, img.split('_')[-1]), aligned_face)
    return aligned_face, landmark

def rect_to_bbox(rect):
    """rect to bbox"""
    # print(rect)
    x = rect[3]
    y = rect[0]
    w = rect[1] - x
    h = rect[2] - y
    return (x, y, w, h)


class FaceMasker:
    def __init__(self, random_mask = True):
        MASKS = ["masks/mask1.png","masks/mask2.png","masks/mask3.png","masks/mask4.png","masks/mask5.png","masks/mask6.png"]
        if random_mask:
            self.mask_path = random.choice(MASKS)
        self.landmark = None
        self.img = None

    def mask(self, img, landmark):
        self.landmark = landmark
        self._face_img = Image.fromarray(img)
        self._mask_img = Image.open(self.mask_path)
        self._mask_face()

    def _mask_face(self):
        nose_bridge = self.landmark[27:31]
        nose_point = nose_bridge[len(nose_bridge) * 1 // 4]
        nose_v = np.array(nose_point)

        chin = self.landmark[0:17]
        chin_len = len(chin)
        chin_bottom_point = chin[chin_len // 2]
        chin_bottom_v = np.array(chin_bottom_point)
        chin_left_point = chin[chin_len // 8]
        chin_right_point = chin[chin_len * 7 // 8]

        # split mask and resize
        width = self._mask_img.width
        height = self._mask_img.height
        width_ratio = 1.2
        new_height = int(np.linalg.norm(nose_v - chin_bottom_v))
        if new_height <= 0:
            new_height = 1

        # left
        mask_left_img = self._mask_img.crop((0, 0, width // 2, height))
        mask_left_width = self.get_distance_from_point_to_line(chin_left_point, nose_point, chin_bottom_point)
        mask_left_width = int(mask_left_width * width_ratio)
        if mask_left_width <= 0:
            mask_left_width = 1
        mask_left_img = mask_left_img.resize((mask_left_width, new_height))

        # right
        mask_right_img = self._mask_img.crop((width // 2, 0, width, height))
        mask_right_width = self.get_distance_from_point_to_line(chin_right_point, nose_point, chin_bottom_point)
        mask_right_width = int(mask_right_width * width_ratio)
        if mask_right_width <= 0:
            mask_right_width = 1
        mask_right_img = mask_right_img.resize((mask_right_width, new_height))

        # merge mask
        size = (mask_left_img.width + mask_right_img.width, new_height)
        mask_img = Image.new('RGBA', size)
        mask_img.paste(mask_left_img, (0, 0), mask_left_img)
        mask_img.paste(mask_right_img, (mask_left_img.width, 0), mask_right_img)

        # rotate mask
        angle = np.arctan2(chin_bottom_point[1] - nose_point[1], chin_bottom_point[0] - nose_point[0])
        rotated_mask_img = mask_img.rotate(angle, expand=True)

        # calculate mask location
        center_x = (nose_point[0] + chin_bottom_point[0]) // 2
        center_y = (nose_point[1] + chin_bottom_point[1]) // 2

        offset = mask_img.width // 2 - mask_left_img.width
        radian = angle * np.pi / 180
        box_x = center_x + int(offset * np.cos(radian)) - rotated_mask_img.width // 2
        box_y = center_y + int(offset * np.sin(radian)) - rotated_mask_img.height // 2

        # add mask
        self._face_img.paste(mask_img, (int(box_x), int(box_y)), mask_img)


    @staticmethod
    def get_distance_from_point_to_line(point, line_point1, line_point2):
        distance = np.abs((line_point2[1] - line_point1[1]) * point[0] +
                          (line_point1[0] - line_point2[0]) * point[1] +
                          (line_point2[0] - line_point1[0]) * line_point1[1] +
                          (line_point1[1] - line_point2[1]) * line_point1[0]) / \
                   np.sqrt((line_point2[1] - line_point1[1]) * (line_point2[1] - line_point1[1]) +
                           (line_point1[0] - line_point2[0]) * (line_point1[0] - line_point2[0]))
        return int(distance)


if __name__ == '__main__':

    fa = FaceAligner(desiredLeftEye=DESIRED_LEFT_EYE_LOC, desiredFaceWidth=FACE_SIZE)
    fm = FaceMasker()

    print("STATUS:    Loading models ...")
    print("STATUS:    Loading detection model ...")
    print(f"STATUS:    Mode: {DETECTION_MODEL_TYPE}")
    print(f"STATUS:    INPUT SIZE: {DETECTION_INPUT_SIZE}")
    print("STATUS:    CHECKING TEST DEVICE ...")
    using_gpu = False
    if TEST_DEVICE != "cpu":
        cuda_available = torch.cuda.is_available()
        if not cuda_available:
            print('gpu device not available, switch to cpu')
            device = torch.device("cpu")
        else:
            normal_recog_net.cuda()
            masked_recog_net.cuda()
            device = torch.device(TEST_DEVICE)
            using_gpu = True
    else:
        device = torch.device("cpu")
    print(f"STATUS:    USING DEVICE: {device}")

    from vision.ssd.config.fd_config import ImageConfiguration
    from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
    from vision.utils.misc import Timer
    class_names = [name.strip() for name in open(DETECTION_LABEL).readlines()]
    num_classes = len(class_names)
    if DETECTION_MODEL_TYPE == "fast":
        model_path = DETECTION_FAST_MODEL_PATH
    elif DETECTION_MODEL_TYPE == "faster":
        model_path = DETECTION_FASTER_MODEL_PATH
    elif DETECTION_MODEL_TYPE == "hybrid":
        model_path = DETECTION_HYBRID_MODEL_PATH
    else:
        print("The model type is wrong!")
        sys.exit(1)
    img_config = ImageConfiguration(DETECTION_INPUT_SIZE)
    det_net = create_Mb_Tiny_RFB_fd(img_config, len(class_names), is_test=True, device=TEST_DEVICE)
    det_predictor = create_Mb_Tiny_RFB_fd_predictor(img_config, det_net, candidate_size=DETECTION_CANDIDATE_SIZE, device=TEST_DEVICE)
    det_net.load(model_path)

    print("STATUS:    Loading landmark model ...")
    landmark_predictor = landmark_detector(model_path=LANDMARKS_MODEL_PATH, test_device=device)

    # load recognition model
    print("STATUS:    Loading normal recognition model ...")
    print("STATUS:    Loading masked recognition model ...")

    normal_recog_net = mfn.MobileFacenet()
    masked_recog_net = mfn.MobileFacenet()


    ckpt = torch.load(RECOGNITION_NORMAL_MODEL_PATH, map_location=device)
    normal_recog_net.load_state_dict(ckpt['net_state_dict'])
    normal_recog_net.eval()

    ckpt = torch.load(RECOGNITION_MASKED_MODEL_PATH, map_location=device)
    masked_recog_net.load_state_dict(ckpt['net_state_dict'])
    masked_recog_net.eval()

    torch.no_grad()
    if using_gpu:
        torch.cuda.synchronize()

    cap = cv2.VideoCapture(0)

    while True:
        _, orig_img = cap.read()

        img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

        # detect face
        boxes, labels, probs = det_predictor.predict(img, DETECTION_CANDIDATE_SIZE / 2, DETECTION_THRESHOLD)
        # get landmark  for wearing mask
        landmark, angle = landmark_predictor.detect(orig_img, boxes[0, :].numpy())

        # wear mask
        fm.mask(img, landmark)
        # get masked and convert to bgr
        masked_img = cv2.cvtColor(np.array(fm._face_img), cv2.COLOR_RGB2BGR)

        for j in range(68):
            cv2.circle(masked_img, (int(landmark[j, 0]), int(landmark[j, 1])), 1, (255, 0, 0))

        aligned = fa.align(img, landmark)

        cv2.imshow("img", masked_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()




