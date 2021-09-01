# -*- coding: utf-8 -*-
# @Author: yirui
# @Date:   2021-09-01 17:39:28
# @Last Modified by:   yirui
# @Last Modified time: 2021-09-01 17:41:42

import torch
import cv2
from config import *
from core import model as mfn

net = mfn.MobileFacenet()
ckpt = torch.load(RECOGNITION_NORMAL_MODEL_PATH, map_location='cpu')
net.load_state_dict(ckpt['net_state_dict'])
net.eval()

print(net)

output_path = "models/recognition/mfn_112_96.onnx"
dummy_input = torch.randn(1, 3, 112, 96).to(torch.device("cpu"))
torch.onnx.export(net, dummy_input, output_path, verbose=False, input_names=['input'], output_names=['classes'])
