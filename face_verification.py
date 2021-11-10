# -*- coding: utf-8 -*-
# @Author: yirui
# @Date:   2021-09-13 15:42:15
# @Last Modified by:   yirui
# @Last Modified time: 2021-09-16 20:27:46

import argparse
import cv2
import time
import torch
import numpy as np
import torch.utils.data
from vision.ssd.config.fd_config import define_img_size
from core import model as mfn
from core.utils import *
from config import *

parser = argparse.ArgumentParser(description='Face Verification for KYC')
parser.add_argument('-s', '--sample', type=str,
                    help='image file name for testing')
parser.add_argument('-t', '--target', type=str,
                    help='image file name used as target')
parser.add_argument('--uid', type=str,
                    help='user id reserved')

args = parser.parse_args()

print("STATUS:    Loading models ...")
print("STATUS:    Loading detection model ...")
print(f"STATUS:    Mode: {DETECTION_MODEL_TYPE}")
print(f"STATUS:    INPUT SIZE: {DETECTION_INPUT_SIZE}")
print("STATUS:    CHECKING TEST DEVICE ...")

# cpu only
using_gpu = False
device = torch.device("cpu")
print(f"STATUS:    USING DEVICE: {device}")


# load detection model
print(f"STATUS:    LOADING DETECTION MODEL ...")
define_img_size(DETECTION_INPUT_SIZE)  # must put define_img_size() before 'import create_mb_tiny_fd, create_mb_tiny_fd_predictor'

from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
class_names = [name.strip() for name in open(DETECTION_LABEL).readlines()]
num_classes = len(class_names)
model_path = DETECTION_FAST_MODEL_PATH
det_net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=device)
det_predictor = create_Mb_Tiny_RFB_fd_predictor(det_net, candidate_size=DETECTION_CANDIDATE_SIZE, device=device)
det_predictor.load(model_path)


print("STATUS:    Loading RECOGNITION MODEL ...")
normal_recog_net = mfn.MobileFacenet()
ckpt = torch.load(RECOGNITION_NORMAL_MODEL_PATH, map_location=device)
normal_recog_net.load_state_dict(ckpt['net_state_dict'])
normal_recog_net.eval()
torch.no_grad()


img = cv2.imread(args.sample)
tar = cv2.imread(args.target)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w, _ = img.shape
boxes, labels, probs = det_predictor.predict(img, DETECTION_CANDIDATE_SIZE / 2, DETECTION_THRESHOLD)
assert boxes.size(0) == 1, "multiple faces detected, please retake"

tar = cv2.cvtColor(tar, cv2.COLOR_BGR2RGB)
tar_boxes, tar_labels, tar_probs = det_predictor.predict(tar, DETECTION_CANDIDATE_SIZE / 2, DETECTION_THRESHOLD)


box = boxes[0,:]
x1, y1, x2, y2 = pos_box(box)
img_crop = img[y1:y2, x1:x2]

tar_box = tar_boxes[0,:]
x1, y1, x2, y2 = pos_box(tar_box)
tar_crop = tar[y1:y2, x1:x2]

# flipped
input_dataset = ImageData([tar_crop, img_crop])
images_loader = torch.utils.data.DataLoader(
    input_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    drop_last=False)

for data in images_loader:
    res = []
    t0 = time.time()
    for d in data:
        res.append(normal_recog_net(d).data.cpu().numpy())
    features = np.concatenate((res[0], res[1]), 1)
    features = features / np.expand_dims(np.sqrt(np.sum(np.power(features, 2), 1)), 1)


scores = np.sum(np.multiply(np.array([features[0]]), np.array([features[1]])), 1)
print(scores)
if scores[0] < VERIFICATION_TRHESHOLD:
    print("identify not matched")
else:
    print('identity matched')
