# -*- coding: utf-8 -*-
# @Author: yirui
# @Date:   2021-09-01 17:39:28
# @Last Modified by:   yirui
# @Last Modified time: 2021-09-01 17:41:36

import torch
import cv2
from core.mask_slim import Slim
import numpy as np

model = Slim()
model.load_state_dict(torch.load(open("models/detection/mask.pth", "rb"), map_location='cpu'))
model.eval()

output_path = "models/detection/mask_64.onnx"
dummy_input = torch.randn(1,3,64,64).to(torch.device("cpu"))
torch.onnx.export(model, dummy_input, output_path, verbose=False, input_names=['input'], output_names=['classes'])

