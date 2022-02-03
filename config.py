# -*- coding: utf-8 -*-
# @Author: yirui
# @Date:   2021-09-01 17:39:28
# @Last Modified by:   yirui
# @Last Modified time: 2021-11-15 20:31:34
from __future__ import annotations
import torch
from vision.kycdeepface import EmbeddingSetup
from vision.ssd.config.fd_config import ImageConfiguration

TEST_DEVICE = 'cpu' # cpu/cuda:0

CAM = True # if cam true, recognize from webcam
CAM_INDEX = 0 # 0,1,2 ... for camera indexs
INPUT_VIDEO_PATH = "test/George.mp4" # input video to test

# MODEL_MODE = "onnx" # onnx/pytorch switch between onnx and pytorch models

DETECTION_LABEL = './models/detection/labels.txt'
DETECTION_FAST_MODEL_PATH = "models/detection/fast.pth"
DETECTION_FASTER_MODEL_PATH = "models/detection/faster.pth"
DETECTION_HYBRID_MODEL_PATH = "models/detection/hybrid.pth"
DETECTION_INPUT_SIZE = 320 # 128/160/320/480/640/1280
DETECTION_MODEL_TYPE = "hybrid" # fast/faster/hybrid
DETECTION_THRESHOLD = 0.7 # face detection threshold
DETECTION_CANDIDATE_SIZE = 1500 # do not modify this if necessary, used for nms candidates
FACE_SIZE = 112 # face size config for alignment, not suggested to change
DESIRED_LEFT_EYE_LOC = (0.3, 0.3) # desired aligned crop size, not suggested to change

# MTCNN detector
MIN_FACE_SIZE = 20
MTCNN_DETECTION_MODEL_PATH = "models/detection/mtcnn_weights.npy"

ENABLE_MASK_DETECTION = True # enable wear mask detection during recognition
ENABLE_LANDMARK_DETECTION = True # enable facial landmark detection during recognition
ENABLE_SHOW_ANGLE = True # shoe angle information on the image, only available when landmark detection is enabled

# RECOGNITION_MODEL_TYPE = "normal" # normal/masked\
ALLOW_BLURRY_FILTERING = False # allow burry filtering to filter out blurry faces for recognition
BLURRY_THRESHOLD = 150 # blurrying filter threshold, can modify based on real case scenario

# model and data path
RECOGNITION_NORMAL_MODEL_PATH = "./models/recognition/mfn.pth"
RECOGNITION_MASKED_MODEL_PATH = "./models/recognition/055.pth"
NORMAL_REGISTERED_EMBEDDING = "./static/normal/registered_data.mat"
NORMAL_REGISTERED_NAME_LIST = "./static/normal/name_list.json"
MASKED_REGISTERED_EMBEDDING = "./static/masked/registered_data.mat"
MASKED_REGISTERED_NAME_LIST = "./static/masked/name_list.json"
LANDMARKS_MODEL_PATH = "./models/detection/landmarks.pth"

NORMAL_RECOGNITION_THRESHOLD = 0.39 # recognition thredhold for clean face, 0-1
MASKED_RECOGNITION_THRESHOLD = 0.397 # recognition thredhold for masked face: 0-1

VERIFICATION_TRHESHOLD = 0.37

SHOW_DETECTION_SIZE = False # show detection bounding box sizes

embedding_setup = EmbeddingSetup(
    device = torch.device("cpu"),
    detection_label = DETECTION_LABEL,
    detection_model_path = DETECTION_FAST_MODEL_PATH,
    detection_candidate_size = DETECTION_CANDIDATE_SIZE,
    detection_threshold = DETECTION_THRESHOLD,
    image_config = ImageConfiguration(DETECTION_INPUT_SIZE),
    embedding_size = (96, 112),
    recogntion_model_path = RECOGNITION_NORMAL_MODEL_PATH,
    desired_left_eye_loc = DESIRED_LEFT_EYE_LOC,
    landmarks_model_path = LANDMARKS_MODEL_PATH
)


# ONNX_DETECTION_MODEL_PATH = "models/detection/onnx_1.onnx"
# ONNX_RECOGNITION_NORMAL_MODEL_PATH = "./models/recognition/onnx.mfn.onnx"
# ONNX_RECOGNITION_MASKED_MODEL_PATH = "./models/recognition/055.onnx"
# ONNX_MASK_DETECTION_PATH = "models/detection/slim_64_latest_onnx.onnx"
# ONNX_LANDMARK_ANGLE_DETECTION_PATH = "models/detection/slim_160_latest_onnx.onnx"
