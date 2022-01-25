# -*- coding: utf-8 -*-
# @Author: yirui
# @Date:   2021-09-01 17:39:28
# @Last Modified by:   yirui
# @Last Modified time: 2021-09-01 17:42:34

import argparse
import os
import sys
import cv2
import json
import time
import torch
import imageio
import scipy.io
import numpy as np
import torch.utils.data
from core import model as mfn
from core.utils import *
from config import *

from wear_mask import FaceMasker
from alignment import FaceAligner
from vision.landmark_detector import Detector as landmark_detector

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

print(f"STATUS:    INITIATE FACE MASKER")
fm = FaceMasker()

print(f"STATUS:    INITIATE FACE ALIGNER")
fa = FaceAligner(desiredLeftEye=DESIRED_LEFT_EYE_LOC, desiredFaceWidth=FACE_SIZE)

# load detection model
from vision.ssd.config.fd_config import ImageConfiguration
config = ImageConfiguration(DETECTION_INPUT_SIZE)
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
det_net = create_Mb_Tiny_RFB_fd(config, len(class_names), is_test=True, device=TEST_DEVICE)
det_predictor = create_Mb_Tiny_RFB_fd_predictor(config, det_net, candidate_size=DETECTION_CANDIDATE_SIZE, device=TEST_DEVICE)
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

def face_capture(img):
    orig_img = cv2.imread(img)
    image = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    boxes, labels, probs = det_predictor.predict(image, DETECTION_CANDIDATE_SIZE / 2, DETECTION_THRESHOLD)

    # no face is detected
    if boxes.size(0) == 0:
        return None, None, None

    # assume that when registering, only one face would be detected.
    box = boxes[0, :]
    x1, y1, x2, y2 = pos_box(box)

    # get face landmark
    landmark, _ = landmark_predictor.detect(orig_img, box.numpy())

    # get aligned face
    aligned_face = fa.align(image, landmark)

    # get masked face
    fm.mask(image, landmark)
    masked_image = np.array(fm._face_img)
    masked_face = fa.align(masked_image, landmark)

    # temp_check = './temp'
    # cv2.imwrite(os.path.join(temp_check, img.split('_')[-1]), aligned_face)
    return aligned_face, masked_face, landmark

def save_features(stored_imgs, name):
    # Load the features from stored imgs(Registered data.)
    img_list = [{'img':os.path.join(stored_imgs,f),'name':name, "registered": "true"} for f in os.listdir(stored_imgs)]

    cropped = []
    masked = []
    landmarks = []

    for i in img_list:
        crop_face, masked_face, landmark = face_capture(i['img'])
        # append to lists when detected face
        if crop_face is not None:
            print(f"{i['img']} - succeed")
            cropped.append(crop_face)
            masked.append(masked_face)
            landmarks.append(landmark)
        else:
            print(f"{i['img']} - failed")
            i['registered'] = "false"

    # save normal
    # t0 = time.time()
    img_dataset = ImageData(cropped)
    images_loader = torch.utils.data.DataLoader(img_dataset, batch_size=32, shuffle=False, num_workers=0, drop_last=False)
    features = None
    for data in images_loader:
        if using_gpu:
            for i in range(len(data)):
                data[i] = data[i].cuda()
        res = [normal_recog_net(d).data.cpu().numpy() for d in data]
        feature = np.concatenate((res[0], res[1]), 1)
        if features is None:
            features = feature
        else:
            features = np.concatenate((features, feature), 0)
    result = {'f':features}
    if os.path.isfile(NORMAL_REGISTERED_EMBEDDING):
        res = load_features(NORMAL_REGISTERED_EMBEDDING)
        f = np.concatenate((result['f'], res['f']),0)
        result = {'f':f}
    if os.path.isfile(NORMAL_REGISTERED_NAME_LIST):
        with open (NORMAL_REGISTERED_NAME_LIST,'r') as f:
            nl = json.load(f)
        img_list.extend(nl['data'])

    with open (NORMAL_REGISTERED_NAME_LIST,'w') as f:
        json.dump({'data':img_list},f)
    scipy.io.savemat(NORMAL_REGISTERED_EMBEDDING, result)

    # save masked
    img_dataset = ImageData(masked)
    images_loader = torch.utils.data.DataLoader(img_dataset, batch_size=32, shuffle=False, num_workers=0, drop_last=False)
    features = None
    for data in images_loader:
        if using_gpu:
            for i in range(len(data)):
                data[i] = data[i].cuda()
        res = [masked_recog_net(d).data.cpu().numpy() for d in data]
        feature = np.concatenate((res[0], res[1]), 1)
        if features is None:
            features = feature
        else:
            features = np.concatenate((features, feature), 0)
    result = {'f':features}
    if os.path.isfile(MASKED_REGISTERED_EMBEDDING):
        res = load_features(MASKED_REGISTERED_EMBEDDING)
        f = np.concatenate((result['f'], res['f']),0)
        result = {'f':f}
    if os.path.isfile(MASKED_REGISTERED_NAME_LIST):
        with open (MASKED_REGISTERED_NAME_LIST,'r') as f:
            nl = json.load(f)
        img_list.extend(nl['data'])
    with open (MASKED_REGISTERED_NAME_LIST,'w') as f:
        json.dump({'data':img_list},f)
    scipy.io.savemat(MASKED_REGISTERED_EMBEDDING, result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='face recognition')

    parser.add_argument('--img_path', default="./stored/tester", type=str,
                        help='The path for stored/registered images, for one person put in the same folder.')
    parser.add_argument('--name', default="tester", type=str,
                        help='The name of registered person')
    parser.add_argument('--root_folder', default='', type=str,
                        help="provide root folder for all face folders")

    args = parser.parse_args()

    if args.root_folder != "":
        names = os.listdir(args.root_folder)
        for name in names:
            save_features(os.path.join(args.root_folder, name), name)
    else:
        save_features(args.img_path, args.name)



