from __future__ import annotations
from vision.utils.lang import lazy
import numpy as np
import math
import torch

def generate_priors(width, height, min_boxes: list[(int, list[int])], clamp=True) -> torch.Tensor:
    priors = []
    for pixels_per_level, min_box_list in min_boxes:
        scale_w = math.ceil(width / pixels_per_level)
        scale_h = math.ceil(height / pixels_per_level)
        for j in range(0, scale_h):
            y_center = (j + 0.5) / scale_h
            for i in range(0, scale_w):
                x_center = (i + 0.5) / scale_w
                for min_box in min_box_list:
                    w = min_box / width
                    h = min_box / height
                    priors.append([
                        x_center,
                        y_center,
                        w,
                        h
                    ])

    priors = torch.tensor(priors)
    if clamp:
        torch.clamp(priors, 0.0, 1.0, out=priors)
    return priors

class ImageConfiguration:
    def __init__(self,
        width,
        height = None,
        ratio = 4/3,
        image_mean = np.array([127, 127, 127]),
        image_std = 128.0,
        iou_threshold = 0.3,
        center_variance = 0.1,
        size_variance = 0.2,
        min_box_setup = [
            # TODO: where do these defaults come from?
            (8, [10, 16, 24]),
            (16, [32, 48]),
            (32, [64, 96]),
            (64, [128, 192, 256])
        ]
    ):
        self.width = width
        if height is None:
            height = int(width / ratio)
        self.height = height
        self.image_mean = image_mean
        self.image_std = image_std
        self.iou_threshold = iou_threshold
        self.center_variance = center_variance
        self.size_variance = size_variance
        self._min_boxes = lazy(lambda: [min_box_list for _, min_box_list in min_box_setup])
        self._image_size = lazy(lambda: [self.width, self.height])
        self._priors = lazy(lambda: generate_priors(width, height, min_box_setup))

    @property
    def image_size(self):
        return self._image_size()

    @property
    def priors(self):
        return self._priors()

    @property
    def min_boxes(self):
        return self._min_boxes()
