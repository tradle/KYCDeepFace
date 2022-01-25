# -*- coding: utf-8 -*-
# @Author: yirui
# @Date:   2021-09-01 17:39:28
# @Last Modified by:   yirui
# @Last Modified time: 2021-11-15 20:52:05
import os
import cv2
import imageio
import torch
import scipy.io
import logging
import numpy as np

def init_log(output_dir):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y%m%d-%H:%M:%S',
                        filename=os.path.join(output_dir, 'log.log'),
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    return logging

def assert_bgr (img):
    if len(img.shape) == 2:
        return np.stack([img]*3,2) # greyscale image -> bgr
    return img

def normalize_image (img, dsize=(96,112)):
    img = cv2.resize(img, dsize)
    return assert_bgr(img)

def image_to_tensor (img):
    """
    Makes sure that the image is turned into a normalized 96x112 pixel rgb image
    with each color channel transformed from a 255 rgb value to a normalized 1.0 ~ -1.0 value

    Parameters
    ----------
    img : cv2 image

    Returns
    -------
    float tensor with numbers from 0.99609375 ... -0.99609375
    """
    # kw: "Gradient Vanishing" prefers not-straight numbers (no straight 1.0 or -1.0)
    img = (img - 127.5) / 128.0 # 0.99609375 ... -0.99609375 (normalized)
    img = img.transpose(2, 0, 1) #  Transpose it into torch order (CHW)
    return torch.from_numpy(img).float()

def image_to_tensors (img):
    """
    Parameters
    ----------
    img : cv2 image

    Returns
    -------
    two tensors with numbers from 0.99609375 ... -0.99609375 - one horizontally flipped
    """
    return (
        image_to_tensor(img),
        image_to_tensor(flip_horz(img))
    )

def flip_horz (img):
    # https://blog.csdn.net/qq_36338754/article/details/104395884 (flipped horizontally)
    return img[:, ::-1, :]

class ImageData(object):
    def __init__(self, img_list):
        img_list = [entry for entry in img_list]
        self.img_list = img_list
        self.len = len(self.img_list)

    def __getitem__(self, index):
        return [entry for entry in image_to_tensors(normalize_image(self.img_list[index]))]

    def __len__(self):
        return self.len

def load_features(path):
    # the path of registered face features
    result = scipy.io.loadmat(path)
    return result

def pos(num):
    n = int(num)
    if n < 0:
        n =0
    return n

def pos_tuple(tp):
    return (pos(tp[0]), pos(tp[1]))

def pos_box(box):
    x1, y1, x2, y2 = box
    return pos(x1), pos(y1), pos(x2 + 0.5), pos(y2 + 0.5)

def getLapVar(image):
    lap_scores = cv2.Laplacian(image, cv2.CV_64F)
    return lap_scores.var()

def xywh_xyxy(box):
    x1 = box[0]
    y1 = box[1]
    x2 = x1+box[2]
    y2 = y1+box[3]
    return [x1, y1, x2, y2]
