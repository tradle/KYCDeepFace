# -*- coding: utf-8 -*-
# @Author: yirui
# @Date:   2021-09-01 17:39:28
# @Last Modified by:   yirui
# @Last Modified time: 2021-09-01 17:42:08
import torch
import cv2
from core.face_landmarks import FaceLandmarks
from core.slim import Slim
import numpy as np

class Detector:
    def __init__(self, model_path, detection_size=(160, 160), test_device="cpu"):
        self.model = Slim()
        model = open(model_path, "rb")
        self.model.load_state_dict(torch.load(model, map_location=test_device))
        model.close()
        self.model.eval()
        self.detection_size = detection_size

    def crop_image(self, orig, bbox):
        bbox = bbox.copy()
        image = orig.copy()
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        face_width = (1 + 2 * 0.25) * bbox_width
        face_height = (1 + 2 * 0.25) * bbox_height
        center = [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]
        bbox[0] = max(0, center[0] - face_width // 2)
        bbox[1] = max(0, center[1] - face_height // 2)
        bbox[2] = min(image.shape[1], center[0] + face_width // 2)
        bbox[3] = min(image.shape[0], center[1] + face_height // 2)
        bbox = bbox.astype(np.int32)
        crop_image = image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        h, w, _ = crop_image.shape
        crop_image = cv2.resize(crop_image, self.detection_size)
        return crop_image, ([h, w, bbox[1], bbox[0]])

    def detect(self, img, bbox):
        crop_image, detail = self.crop_image(img, bbox)
        crop_image = (crop_image - 127.0) / 127.0
        crop_image = np.array([np.transpose(crop_image, (2, 0, 1))])
        crop_image = torch.tensor(crop_image).float()
        with torch.no_grad():
            raw = self.model(crop_image)[0].cpu().numpy()
            landmark = raw[0:136].reshape((-1, 2))
        landmark[:, 0] = landmark[:, 0] * detail[1] + detail[3]
        landmark[:, 1] = landmark[:, 1] * detail[0] + detail[2]
        h, w, _ = img.shape
        return FaceLandmarks(landmark, (w, h), bbox)
