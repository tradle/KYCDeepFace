# -*- coding: utf-8 -*-
# @Author: yirui
# @Date:   2021-09-13 15:42:15
# @Last Modified by:   yirui
# @Last Modified time: 2021-11-18 00:38:48

import argparse
import cv2
import time
import torch
import numpy as np
import torch.utils.data
from core import model as mfn
from core.utils import *
from config import *
from mtcnn import MTCNN
from mtcnn_alignment import FaceAligner

using_gpu = False
device = torch.device("cpu")

parser = argparse.ArgumentParser(description='input for sample and target images')
parser.add_argument('--target', type=str,
                    help='test against target file')
parser.add_argument('--sample', type=str,
                    help='sample image to test')

args = parser.parse_args()

# from vision.ssd.config.fd_config import ImageConfiguration
# from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
# 
# config = ImageConfiguration(
#     # DETECTION_INPUT_SIZE
#     320
# )
# class_names = [name.strip() for name in open(
#     # DETECTION_LABEL
#     './models/detection/labels.txt'
# ).readlines()]
# num_classes = len(class_names)
# det_net = create_Mb_Tiny_RFB_fd(
#     config,
#     num_classes,
#     is_test=True,
#     device=device
# )
# det_predictor = create_Mb_Tiny_RFB_fd_predictor(
#     config,
#     det_net,
#     # candidate_size=DETECTION_CANDIDATE_SIZE,
#     candidate_size=1500,
#     device=device
# )
# det_predictor.load(
#     # DETECTION_FAST_MODEL_PATH
#     "./models/detection/fast.pth"
# )

det_predictor = MTCNN(weights_file=MTCNN_DETECTION_MODEL_PATH, min_face_size=MIN_FACE_SIZE)


print("STATUS:    Loading RECOGNITION MODEL ...")
normal_recog_net = mfn.MobileFacenet()
ckpt = torch.load(
    # RECOGNITION_NORMAL_MODEL_PATH
    './models/recognition/mfn.pth'
, map_location=device)
normal_recog_net.load_state_dict(ckpt['net_state_dict'])
normal_recog_net.eval()
torch.no_grad()

def process_image (img):
    data = cv2.imread(img)
    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    # boxes, labels, probs = det_predictor.predict(data, DETECTION_CANDIDATE_SIZE / 2, DETECTION_THRESHOLD)
    results = det_predictor.detect_faces(data)
    assert len(results) == 1, "multiple faces detected, please retake"
    box = results[0]['box']
    x1, y1, x2, y2 = xywh_xyxy(box)
    cropped = data[y1:y2, x1:x2]
    return cropped

input_dataset = ImageData([
    process_image(args.target),
    process_image(args.sample)
])
images_loader = torch.utils.data.DataLoader(
    input_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0,
    drop_last=False)

featureLs = None
featureRs = None

for data in images_loader: # these are 2x2 images [[target, horz_flipped(target)], [sample, horz_flipped(sample)]]!
    res = [normal_recog_net(d).data.cpu().numpy() for d in data]
    print(len(res))

    if featureLs is None:
        featureLs = res[0]
    else:
        featureLs = np.concatenate((featureLs, res[0]), 0)
    if featureRs is None:
        featureRs = res[1]
    else:
        featureRs = np.concatenate((featureRs, res[1]), 0)

# res[0] contain normal and flipped embedding for sample
# res[1] contain normal and flipped embedding for target

# features = np.concatenate((res[0], res[1]), 1)
mu = np.mean(np.concatenate((featureLs, featureRs), 0), 0)
mu = np.expand_dims(mu, 0)
featureLs = featureLs - mu
featureRs = featureRs - mu
featureLs = featureLs / np.expand_dims(np.sqrt(np.sum(np.power(featureLs, 2), 1)), 1)
featureRs = featureRs / np.expand_dims(np.sqrt(np.sum(np.power(featureRs, 2), 1)), 1)

scores = np.sum(np.multiply(featureLs, featureRs), 1)
# print(featureL.shape)


# scores = np.sum(np.multiply(featureL, featureR), 1)
print(scores)

if np.max(scores) < VERIFICATION_TRHESHOLD:
    print("identify not matched")
else:
    print('identity matched')
