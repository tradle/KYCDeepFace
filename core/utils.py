import os
import cv2
import imageio
import torch
import scipy.io
import logging

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

class ImageData(object):
    def __init__(self, img_list):
        self.img_list = img_list

    def __getitem__(self, index):
        # img = imageio.imread(self.img_list[index])
        img = self.img_list[index]
        img = cv2.resize(img, (96,112))
        if len(img.shape) == 2:
            img = np.stack([img]*3,2)
        imglist = [img, img[:, ::-1, :]]
        for i in range(len(imglist)):
            imglist[i] = (imglist[i] - 127.5) / 128.0
            imglist[i] = imglist[i].transpose(2, 0, 1)
        imgs = [torch.from_numpy(i).float() for i in imglist]
        return imgs

    def __len__(self):
        return len(self.img_list)

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
    return pos(x1), pos(y1), pos(x2), pos(y2)

def getLapVar(image):
    lap_scores = cv2.Laplacian(image, cv2.CV_64F)
    return lap_scores.var()
