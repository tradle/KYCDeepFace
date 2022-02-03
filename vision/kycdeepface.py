from __future__ import annotations
import cv2
import numpy as np
import torch
from torch.functional import Tensor

from torch.utils.data import DataLoader
from base64 import b64encode, urlsafe_b64encode, urlsafe_b64decode
from core import model as mfn
from core.face_landmarks import format_landmarks
from core.utils import image_to_tensors
from vision.landmark_detector import Detector as LandmarkDetector
from vision.face_aligner import FaceAligner
from vision.utils.lang import lazy_timed, timed_iter, timed, to_tuple
from vision.ssd.config.fd_config import ImageConfiguration

def load_det_predictor (device, image_config, detection_label, model_path, candidate_size):
    from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
    class_names = [name.strip() for name in open(detection_label).readlines()]
    det_net = create_Mb_Tiny_RFB_fd(image_config, len(class_names), is_test=True, device=device)
    det_predictor = create_Mb_Tiny_RFB_fd_predictor(image_config, det_net, candidate_size, device=device)
    det_net.load(model_path)
    return det_predictor

def load_normal_recog_net (device, model_path):
    net = mfn.MobileFacenet()
    ckpt = torch.load(model_path, map_location=device)
    net.load_state_dict(ckpt['net_state_dict'])
    net.eval()
    torch.no_grad()
    return net

def load_landmark_detector(model_path: str):
    return LandmarkDetector(model_path=model_path)

def encode_embeddings (embeddings):
    return urlsafe_b64encode(embeddings.tobytes('C')).decode()

def decode_embeddings (urlsafe_b64):
    # https://www.markhneedham.com/blog/2018/04/07/python-serialize-deserialize-numpy-2d-arrays/
    return np.frombuffer(urlsafe_b64decode(urlsafe_b64), dtype='float32').reshape((1, 256))

def format_embeddings (landmarks, image, embeddings, add_image=True):
    # array containing normal embedding and flipped embedding
    result = {
        'landmarks': landmarks,
        'embedding': encode_embeddings(embeddings)
    }
    if add_image:
        result['image'] = data_uri(image),
    return result

def face_match (a, b) -> float:
    features = np.concatenate((a, b), 0)
    features = features / np.expand_dims(np.sqrt(np.sum(np.power(features, 2), 1)), 1)
    scores = np.sum(np.multiply(np.array([features[0]]), np.array([features[1]])), 1)
    return np.clip(np.max(scores), 0.0, 1.0)

class EmbeddingSetup:
    def __init__(self,
        device: torch.device,
        detection_label: str,
        detection_model_path: str,
        detection_candidate_size: int,
        detection_threshold: float,
        image_config: ImageConfiguration,
        recogntion_model_path: str,
        desired_left_eye_loc: tuple((float, float)),
        landmarks_model_path: str,
        embedding_size: tuple(int, int),
        timings: dict[str, float] = {}
    ):
        self.device = device
        self.timings = timings
        self.detection_candidate_size = detection_candidate_size
        self.detection_threshold = detection_threshold
        self._det_predictor = lazy_timed(self.timings, 'load_det_predictor', lambda: load_det_predictor(device, image_config, detection_label, model_path=detection_model_path, candidate_size=detection_candidate_size))
        self._normal_recog_net = lazy_timed(self.timings, 'load_normal_recog_net', lambda: load_normal_recog_net(device, model_path=recogntion_model_path))
        self._landmark_detector = lazy_timed(self.timings, 'load_landmark_detector', lambda: load_landmark_detector(model_path=landmarks_model_path))
        self.fa = FaceAligner(left_eye=desired_left_eye_loc, dsize=embedding_size)

    @property
    def det_predictor(self):
        return self._det_predictor()
    
    @property
    def normal_recog_net(self):
        return self._normal_recog_net()
    
    @property
    def landmark_detector(self):
        return self._landmark_detector()

def embedding (setup: EmbeddingSetup, torch_data):
    return setup.normal_recog_net(torch_data).data.cpu().numpy()

def detect_faces (setup: EmbeddingSetup, image: list[list[list[int, int, int]]]) -> Tensor:
    boxes, labels, probs = setup.det_predictor.predict(image, setup.detection_candidate_size / 2, setup.detection_threshold)
    return boxes

def extract_face (setup: EmbeddingSetup, image: list[list[list[int, int, int]]], box):
    landmarks = setup.landmark_detector.detect(image, box.numpy())
    return [setup.fa.align_face(image, landmarks), format_landmarks(landmarks)]

def extract_faces (setup: EmbeddingSetup, image: list[list[list[int, int, int]]], boxes):
    return [extract_face(setup, image, boxes[i,:]) for i in range(boxes.size(0))]

def data_uri (image: list[list[list[int, int, int]]]):
    b64 = b64encode(cv2.imencode(img=image, ext='.png')[1]).decode()
    return f'data:image/png;base64,{b64}'

def face_embeddings (setup: EmbeddingSetup, image: list[list[list[int, int, int]]]):
    timings = {}
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = timed(timings, 'detect_faces', lambda: detect_faces(setup, image))
    extract_res = timed(timings, 'extract_faces', lambda: extract_faces(setup, image, boxes))
    if len(extract_res) == 0:
        return timings, ()
    face_images, landmarks = list(zip(*extract_res))
    batches = DataLoader(
        [image_to_tensors(image) for image in face_images],
        # The are expected in a batched fashion, but with a batch_size > 1
        # the a batch for the entries [[1, a], [2, b], [3, c]] will be turned int
        # [[1,2,3], [a,b,c]] (num = normal faces; chr = flipped faces). This would
        # mean that all faces will be extracted immediately and that we need to re-
        # combine the entries for the embeddings.
        #
        # TODO: there must be a better solution to this (probably without DataLoader)
        batch_size = 1,
        shuffle = False,
        num_workers = 0
    )
    return timings, (
        (landmarks[index], face_images[index], np.concatenate((embeddings[0], embeddings[1]), 1))
        for index, embeddings in enumerate(to_tuple(embedding(setup, alternating_normal_flipped) for batch in batches for alternating_normal_flipped in batch))
    )
