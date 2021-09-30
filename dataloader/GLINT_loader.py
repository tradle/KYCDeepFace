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
from multiprocessing import Pool
import math
from sklearn.preprocessing import LabelEncoder

class GLINT_Face(object):
    def __init__(self, data_folder):
        # data_folder - absolute path
        self.data_folder = data_folder
        _ids = os.listdir(self.data_folder)
        # _ids = _ids[:200] # for testing only
        image_list = []
        label_list = []
        
        exp = 0
        exp_str = '00:00:00'
        # for i, _id in enumerate(_ids):
        #     t0 = time.time()
        #     if i!=len(_ids)-1:
        #         print(f"INFO:  loading {i}/{len(_ids)}, expected to be finished after {exp_str}", flush=True, end='\r')
        #     else:
        #         print(f"INFO:  loading {i}/{len(_ids)}, expected to be finished after {exp_str}")    
        #     for fn in os.listdir(osp.join(self.data_folder, _id)):
        #         image_list.append(osp.join(self.data_folder, _id, fn))
        #         label_list.append(i)
                
        #     if i%1000 == 0:
        #       t1 = time.time()
        #       exp = (t1-t0)*(len(_ids)-i)
        #       exp_str = str(datetime.timedelta(seconds=exp))
        
        num_process = 20
        print(f'INFO:  init {num_process} workers for data indexing')
        chunk_size = math.floor(len(_ids)/num_process)
        with Pool(processes = num_process) as p:
            data = p.map(self.__prepare__, [_ids[x:x+chunk_size] for x in range(0, len(_ids), chunk_size)])
        
        print('\nINFO:  joining pool result ...')
        temp = []
        for item in data:
            temp.extend(item)
        zip(*[[1,2], [3,4], [5,6]])
        self.image_list, self.label_list = zip(*temp)
        le = LabelEncoder()
        self.label_list = le.fit_transform(self.label_list)
        # self.label_list = label_list
        self.class_nums = len(np.unique(self.label_list))
        
        print(f'Data Summary:  {len(self.image_list)} data indexed with {self.class_nums} identities')

    def __prepare__(self, _ids):
        output = []
        exp = 0
        exp_str = '00:00:00'
        for i, _id in enumerate(_ids):
            t0 = time.time()
            # print(f"INFO:  loading {i}/{len(_ids)}, expected to be finished after {exp_str}", flush=True, end='\x1b[1K\r')
            print(f"INFO:  loading {i}/{len(_ids)}, expected to be finished after {exp_str}", flush=True, end='\r')
            for fn in os.listdir(osp.join(self.data_folder, _id)):
                output.append([osp.join(self.data_folder, _id, fn), _id])
            if i%1000 == 0:
                t1 = time.time()
                exp = (t1-t0)*(len(_ids)-i)
                exp_str = str(datetime.timedelta(seconds=exp))
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
