# -*- coding: utf-8 -*-
# @Author: yirui
# @Date:   2021-09-16 20:27:19
# @Last Modified by:   yirui
# @Last Modified time: 2021-09-16 20:27:29

import numpy as np
import scipy.misc
import os
import torch

class GLINT_Face(object):
    def __init__(self, data_folder):
        # data_folder - absolute path
        self.data_folder = data_folder
        _ids = os.listdir(self.data_folder)

        image_list = []
        label_list = []
        
        for i, _id in enumerate(_ids):
            for fn in _ids:
                image_list.append(os.path(self.data_folder, _id, fn))
                label_list.append(i)

        self.image_list = image_list
        self.label_list = label_list
        self.class_nums = len(np.unique(self.label_list))

    def __getitem__(self, index):
        img_path = self.image_list[index]
        target = self.label_list[index]
        img = scipy.misc.imread(img_path)

        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)
        flip = np.random.choice(2)*2-1
        img = img[:, ::flip, :]
        img = (img - 127.5) / 128.0
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()

        return img, target

    def __len__(self):
        return len(self.image_list)
