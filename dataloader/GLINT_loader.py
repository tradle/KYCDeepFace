# -*- coding: utf-8 -*-
# @Author: yirui
# @Date:   2021-09-16 20:27:19
# @Last Modified by:   yirui
# @Last Modified time: 2021-09-16 20:27:29

import numpy as np
import scipy.misc
from PIL import Image
import os
import torch
import time
import datetime
from os import path as osp
from multiprocessing import Pool, current_process
import math
from sklearn.preprocessing import LabelEncoder

def print_t(text, flush=False, end='\n'):
    print(f"{time.strftime('%H:%M:%S')} INFO:  {text}", flush=flush, end=end)

class GLINT_Face(object):
    def __init__(self, data_folder):
        # data_folder - absolute path
        self.data_folder = data_folder
        _ids = os.listdir(self.data_folder)
        # _ids = _ids[:200] # for testing only
        image_list = []
        label_list = []
        start = time.time()

        num_process = 20

        print_t(f'init {num_process} workers for data indexing of {len(_ids)} images')
        chunk_size = math.floor(len(_ids)/num_process)
        with Pool(processes = num_process) as p:
            data = p.map(self.__prepare__, [_ids[x:x+chunk_size] for x in range(0, len(_ids), chunk_size)])

        print('')
        print_t('joining pool result ...')
        temp = []
        for item in data:
            temp.extend(item)
        # zip(*[[1,2], [3,4], [5,6]])
        self.image_list, self.label_list = zip(*temp)
        le = LabelEncoder()
        self.label_list = le.fit_transform(self.label_list)
        # self.label_list = label_list
        self.class_nums = len(np.unique(self.label_list))

        print_t(f'Data Summary: {len(self.image_list)} data indexed with {self.class_nums} identities')

    def __prepare__(self, _ids):
        output = []
        num_items = len(_ids)
        previous_percent = 0
        is_first_thread = current_process().name == 'ForkPoolWorker-1'
        for i, _id in enumerate(_ids):
            # end = '\x1b[1K\r' # Windows
            end = '\r' # Linux / Mac
            # print(f"INFO:  loading {i}/{len(_ids)}, expected to be finished after {exp_str}", flush=True, end=end)
            for fn in os.listdir(osp.join(self.data_folder, _id)):
                output.append([osp.join(self.data_folder, _id, fn), _id])
            if is_first_thread:
                percent = math.floor(100/num_items*i)
                if percent != previous_percent:
                    previous_percent = percent
                    print_t(f"loading (~{percent}%)", flush=True, end=end)
        return output

    def __getitem__(self, index):
        img_path = self.image_list[index]
        target = self.label_list[index]
        img = Image.open(img_path)
        img = img.resize((96,112))
        img = np.array(img)

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

