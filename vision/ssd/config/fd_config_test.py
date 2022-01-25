from torch.functional import Tensor
import torch
from vision.ssd.config.fd_config import ImageConfiguration
import unittest

class ImageConfigurationTest(unittest.TestCase):

    def test_min_boxes(self):
        conf = ImageConfiguration(320)
        self.assertListEqual(conf.min_boxes, [
            [10, 16, 24],
            [32, 48],
            [64, 96],
            [128, 192, 256]
        ])

    def test_320(self):
        conf = ImageConfiguration(320)
        size, priors = (conf.image_size, conf.priors)
        self.assertListEqual(size, [320, 240])
        self.assertIsInstance(priors, Tensor)
        self.assertEqual(priors.dtype, torch.float32)
        self.assertEqual(priors.shape, torch.Size([4420, 4]))
        self.assertListEqual(priors[0].tolist(), [0.012500000186264515, 0.01666666753590107, 0.03125, 0.0416666679084301])
        self.assertListEqual(priors[453].tolist(), [0.7875000238418579, 0.11666666716337204, 0.03125, 0.0416666679084301])
        self.assertListEqual(priors[2310].tolist(), [0.26249998807907104, 0.6499999761581421, 0.03125, 0.0416666679084301])
        self.assertListEqual(priors[4419].tolist(), [0.8999999761581421, 0.875, 0.800000011920929, 1.0])

    def test_480(self):
        conf = ImageConfiguration(480)
        self.assertListEqual(conf.image_size, [480, 360])
