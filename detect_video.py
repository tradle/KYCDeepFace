# -*- coding: utf-8 -*-
# @Author: yirui
# @Date:   2021-09-01 17:39:28
# @Last Modified by:   yirui
# @Last Modified time: 2021-09-01 17:41:47

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
from vision.ssd.config.fd_config import define_img_size
from core import model as mfn
from core.utils import *
from landmark_detector import Detector as landmark_detector
from mask_predictor import Detector as mask_detector
from config import *
import pickle
from alignment import FaceAligner
import uuid

print("STATUS:    Loading models ...")
print("STATUS:    Loading detection model ...")
print(f"STATUS:    Mode: {DETECTION_MODEL_TYPE}")
print(f"STATUS:    INPUT SIZE: {DETECTION_INPUT_SIZE}")
print("STATUS:    CHECKING TEST DEVICE ...")
using_gpu = False
if TEST_DEVICE != "cpu":
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print('MESSAGE:    GPU device not available, switch to CPU')
        device = torch.device("cpu")
    else:
        normal_recog_net.cuda()
        masked_recog_net.cuda()
        device = torch.device(TEST_DEVICE)
        using_gpu = True
else:
    device = torch.device("cpu")
print(f"STATUS:    USING DEVICE: {device}")

fa = FaceAligner(desiredLeftEye=DESIRED_LEFT_EYE_LOC, desiredFaceWidth=FACE_SIZE)
print(f"STATUS:    LOADING FACE ALIGNER")

# load detection model
define_img_size(DETECTION_INPUT_SIZE)  # must put define_img_size() before 'import create_mb_tiny_fd, create_mb_tiny_fd_predictor'
from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
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
det_net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=device)
det_predictor = create_Mb_Tiny_RFB_fd_predictor(det_net, candidate_size=DETECTION_CANDIDATE_SIZE, device=device)
det_net.load(model_path)

if ENABLE_MASK_DETECTION:
    print("STATUS:    Loading mask model ...")
    mask_predictor = mask_detector(test_device=device)

if ENABLE_LANDMARK_DETECTION:
    print("STATUS:    Loading landmark model ...")
    landmark_predictor = landmark_detector(test_device=device)


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

print("STATUS:    Loading registered data (normal/masked)...")
# load existing features
res = load_features(NORMAL_REGISTERED_EMBEDDING)
normal_features = res['f']
normal_features = normal_features / np.expand_dims(np.sqrt(np.sum(np.power(normal_features, 2), 1)), 1)
with open (NORMAL_REGISTERED_NAME_LIST, 'r') as f:
    normal_name_list = json.load(f)

res = load_features(MASKED_REGISTERED_EMBEDDING)
maskded_features = res['f']
maskded_features = maskded_features / np.expand_dims(np.sqrt(np.sum(np.power(maskded_features, 2), 1)), 1)
with open (MASKED_REGISTERED_NAME_LIST, 'r') as f:
    masked_name_list = json.load(f)

if CAM:
    cap = cv2.VideoCapture(CAM_INDEX)  # capture from camera
else:
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH) # capture from video

timer = Timer()
lap_scores = []
lap_shapes = []
sum = 0
while True:
    ret, orig_image = cap.read()

    if orig_image is None:
        print("end")
        break
    # rotate clock wise
    # orig_image=cv2.transpose(orig_image)
    # orig_image=cv2.flip(orig_image,flipCode=1)

    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image_height, image_width, _ = image.shape
    timer.start()
    boxes, labels, probs = det_predictor.predict(image, DETECTION_CANDIDATE_SIZE / 2, DETECTION_THRESHOLD)

    cropped = []
    names = []
    masked = []
    landmarks = []
    angles = []
    # check whether maksed

    for i in range(boxes.size(0)):
        box = boxes[i, :]
        label = f" {probs[i]:.2f}"

        if SHOW_DETECTION_SIZE:
        # cv2.rectangle(orig_image, (int(box[0])+1, int(box[1])+1), (int(box[2])+1, int(box[3])+1), (225, 255, 255), 2)
            cv2.rectangle(orig_image, pos_tuple((int(box[0])-2, int(box[1])-2)), pos_tuple((int(box[2])+2, int(box[3])-2)), (255, 244, 23), 2)
            cv2.rectangle(orig_image, pos_tuple((int(box[0])-2, int(box[1])+2)), pos_tuple((int(box[2])-2, int(box[3])+2)), (90, 44, 255), 2)
            cv2.rectangle(orig_image, pos_tuple((int(box[0]), int(box[1]))), pos_tuple((int(box[2]), int(box[3]))), (255, 255, 255), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            loc_x = pos(int(box[0]+5))
            loc_y = pos(int(box[1]+20))
            scale = 0.6
            face_w, face_h = [pos(int(box[2]-box[0])), pos(int(box[3]-box[1]))]
            if face_w <100 and face_h<100:
                scale = 0.5
            blk = np.zeros(orig_image.shape, np.uint8)
            cv2.rectangle(blk, pos_tuple((int(box[0]), int(box[1]))), pos_tuple((int(box[2]), int(box[3]))), (125, 125, 125), cv2.FILLED)
            orig_image = cv2.addWeighted(orig_image, 1.0, blk, 0.5, 1)
            # cv2.putText(orig_image, f"{face_w},{face_h}", (loc_x, loc_y), font, scale, (255, 255, 255), 2)
        else:
            cv2.rectangle(orig_image, pos_tuple((int(box[0]), int(box[1]))), pos_tuple((int(box[2]), int(box[3]))), (255, 255, 255), 2)

        x1, y1, x2, y2 = pos_box(box)

        face_crop = image[y1:y2, x1:x2]

        cv2.imwrite(f'test/{str(uuid.uuid4())}.jpg', cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))

        landmark = None

        # calc facial landmarks
        if ENABLE_LANDMARK_DETECTION:
            landmark, angle = landmark_predictor.detect(orig_image, box.numpy())
            landmarks.append(landmark)
            angles.append(angle)

         # check face masked?
        if ENABLE_MASK_DETECTION:
            classes = mask_predictor.detect(orig_image[y1:y2, x1:x2])
            index = classes.argmax()
            if index == 0:
                masked.append(False)
                cropped.append(face_crop)
            else:
                masked.append(True)
                if landmark is None:
                    cropped.append(face_crop)
                else:
                    aligned = fa.align(image, landmark)
                    cropped.append(aligned)
        else:
            masked.append(False)
            cropped.append(face_crop)

        if ALLOW_BLURRY_FILTERING:
            temp_crop = image[y1:y2, x1:x2]
            temp_crop_resized = cv2.resize(temp_crop, (96, int((y2-y1)*(x2-x1)/96)))
            lap_var = getLapVar(cv2.cvtColor(temp_crop, cv2.COLOR_RGB2GRAY))
            if lap_var < BLURRY_THRESHOLD:
                names.append("blurry")
            else:
                names.append("unknown")

    input_dataset = ImageData(cropped)
    images_loader = torch.utils.data.DataLoader(
        input_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        drop_last=False)

    # assume # faces detected in one video is less than 32
    # otherwisw will produce wrong answer
    for data in images_loader:
        if using_gpu:
            for i in range(len(data)):
                data[i] = data[i].cuda()
        res = []
        t0 = time.time()
        # res = [normal_recog_net(d).data.cpu().numpy() for d in data]
        for j, d in enumerate(data):
            try:
                if masked[j]:
                    res.append(masked_recog_net(d).data.cpu().numpy())
                else:
                    res.append(normal_recog_net(d).data.cpu().numpy())
            except:
                res.append(normal_recog_net(d).data.cpu().numpy())
        t1 = time.time()
        print(f'recog calculation: {t1-t0}')
        # print(res)
        featuresR = np.concatenate((res[0], res[1]), 1)
        featuresR = featuresR / np.expand_dims(np.sqrt(np.sum(np.power(featuresR, 2), 1)), 1)

        for j, featureR in enumerate(featuresR):
            if masked[j]:
                scores = np.sum(np.multiply(maskded_features, np.array([featureR])), 1)
                max_score_index = np.argmax(scores)
                max_score = scores[max_score_index]
                name = masked_name_list['data'][max_score_index]['name']
                if max_score < MASKED_RECOGNITION_THRESHOLD:
                    name = 'unknown'
            else:
                scores = np.sum(np.multiply(normal_features, np.array([featureR])), 1)
                max_score_index = np.argmax(scores)
                max_score = scores[max_score_index]
                name = normal_name_list['data'][max_score_index]['name']
                if max_score < NORMAL_RECOGNITION_THRESHOLD:
                    name = 'unknown'

            print(f"potential name {name} - max score {max_score} - masked {masked[j]}")

            if ALLOW_BLURRY_FILTERING:
                if names[i] != "blurry":
                    names[i] = name
                else:
                    names[i] = "blurry"
            else:
                names.append(name)

    interval = timer.end()
    print(f"avg recognition time per face: {interval}")

    print(names)

    for i in range(boxes.size(0)):
        box = boxes[i, :]
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = 0.5
        box_width = box[2] - box[0]
        loc_x = pos(int(box[0]+5+box_width))
        loc_y = pos(int(box[1]+20))

        cv2.putText(orig_image, f"{names[i]}", (loc_x, loc_y), font, scale, (255, 255, 255), 2)

        if ENABLE_MASK_DETECTION:
            cv2.putText(orig_image, f"masked - {masked[i]}", (loc_x, loc_y+15), font, scale, (255, 255, 255), 1)

        if ENABLE_LANDMARK_DETECTION:
            for j in range(68):
                cv2.circle(orig_image, (int(landmarks[i][j, 0]), int(landmarks[i][j, 1])), 1, (255, 0, 0))

        if ENABLE_SHOW_ANGLE:
            cv2.putText(orig_image, f"yaw - {round(angles[i][0], 2)}", (loc_x, loc_y+30), font, scale, (255, 255, 255), 1)
            cv2.putText(orig_image, f"pitch - {round(angles[i][1], 2)}", (loc_x, loc_y+45), font, scale, (255, 255, 255), 1)
            cv2.putText(orig_image, f"roll - {round(angles[i][2], 2)}", (loc_x, loc_y+60), font, scale, (255, 255, 255), 1)


    sum += boxes.size(0)
    cv2.imshow('annotated', orig_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

print("all face num:{}".format(sum))
