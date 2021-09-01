# -*- coding: utf-8 -*-
# @Author: yirui
# @Date:   2021-09-01 17:39:28
# @Last Modified by:   yirui
# @Last Modified time: 2021-09-01 17:41:19

import torch
import cv2
import torch.onnx
from config import *
from vision.ssd.data_preprocessing import PredictionTransform
from vision.ssd.config.fd_config import define_img_size
# input_size = 320
input_size = 640
define_img_size(input_size)
# from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
from vision.ssd.config import fd_config

if DETECTION_MODEL_TYPE == "fast":
    model_path = DETECTION_FAST_MODEL_PATH
    output_path = f"models/detection/fast_{input_size}.onnx"
elif DETECTION_MODEL_TYPE == "faster":
    model_path = DETECTION_FASTER_MODEL_PATH
    output_path = f"models/detection/faster_{input_size}.onnx"
elif DETECTION_MODEL_TYPE == "hybrid":
    model_path = DETECTION_HYBRID_MODEL_PATH
    output_path = f"models/detection/hybrid_{input_size}.onnx"

class_names = [name.strip() for name in open(DETECTION_LABEL).readlines()]
det_net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=torch.device("cpu"))
det_net.load(model_path)
det_net.eval()

# dummy_input = torch.randn(1,3,240,320).to(torch.device("cpu"))
dummy_input = torch.randn(1,3,480,640).to(torch.device("cpu"))
torch.onnx.export(det_net, dummy_input, output_path, verbose=False, input_names=['input'], output_names=['scores', 'boxes'])

# get testing outcome from orig net
# image = cv2.imread('test/testing_img.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# transform = PredictionTransform(fd_config.image_size, fd_config.image_mean_test,fd_config.image_std)
# image = transform(image)
# images = image.unsqueeze(0)

# images = images.to(torch.device("cpu"))

# with torch.no_grad():
#     scores, boxes = det_net.forward(image)

# print(scores, boxes)


# # Export the model
# torch.onnx.export(torch_model,               # model being run
#                   x,                         # model input (or a tuple for multiple inputs)
#                   "super_resolution.onnx",   # where to save the model (can be a file or file-like object)
#                   export_params=True,        # store the trained parameter weights inside the model file
#                   opset_version=10,          # the ONNX version to export the model to
#                   do_constant_folding=True,  # whether to execute constant folding for optimization
#                   input_names = ['input'],   # the model's input names
#                   output_names = ['output'], # the model's output names
#                   dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
#                                 'output' : {0 : 'batch_size'}})
