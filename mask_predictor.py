# -*- coding: utf-8 -*-
# @Author: yirui
# @Date:   2021-09-01 17:39:28
# @Last Modified by:   yirui
# @Last Modified time: 2021-09-01 17:42:22
import torch
import cv2
from core.mask_slim import Slim
import numpy as np
import time


class Detector:
    def __init__(self, detection_size=(64, 64), test_device='cpu'):
        self.model = Slim()
        self.model.load_state_dict(torch.load(open("models/detection/mask.pth", "rb"), map_location=test_device))
        self.model.eval()
        # self.model.cuda()
        self.detection_size = detection_size


    def detect(self, imgs):
        crop_image = cv2.resize(imgs, self.detection_size)
        crop_image = (crop_image - 127.0) / 127.0
        crop_image = np.array([np.transpose(crop_image, (2, 0, 1))])
        crop_image = torch.tensor(crop_image).float()
        with torch.no_grad():
            start = time.time()
            raw = self.model(crop_image)[0].cpu().numpy()
            end = time.time()
        return raw
