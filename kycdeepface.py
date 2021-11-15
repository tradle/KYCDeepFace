# -*- coding: utf-8 -*-
# @Author: yirui
# @Date:   2021-10-06 17:11:26
# @Last Modified by:   yirui
# @Last Modified time: 2021-11-14 20:57:27

# KYCDEEP FACE
# Covers
# [] face registration
# [] model download
# [] face detection
# [] face alignement
# [] blurry filtering
# [] tracking?
# [] db
# [] face recognition
# [] face verification

# Notes
# k means for cluster generation
# store centroid

from vision.ssd.config.fd_config import define_img_size
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
from core import model as mfn
from core.utils import *
import torch
import sqlite3
import os
import ast
import cv2
import pickle
from sklearn.cluster import KMeans
import numpy as np

class KYCDeepFace:
    def __init__(self, det_model_path='models/detection/fast.pth', rec_model_path='models/recognition/mfn.pth', device='cpu'):
        # init detector
        print('INFO:  load face detector')
        class_names = ['bg', 'face']
        define_img_size(320)
        self.det_net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=device)
        self.detector = create_Mb_Tiny_RFB_fd_predictor(self.det_net, candidate_size=1500, device=device)
        self.det_net.load(det_model_path)

        # init recognizer
        print('INFO:  load face recognizer')
        self.recognizer = mfn.MobileFacenet()
        ckpt = torch.load(rec_model_path, map_location=device)
        self.recognizer.load_state_dict(ckpt['net_state_dict'])
        self.recognizer.eval()
        torch.no_grad()

        self.db_optimized = False
        self.centers = False
        if os.path.exists('centroids.pkl'):
            with open('centroids.pkl', 'rb') as f:
                self.centers = pickle.load(f)
                self.db_optimized = True
        else:
            self.db_optimized = False
            print('INFO: db optimization not performed, running in default mode')

        self.db = self.connect_sqlite_db()

    def connect_sqlite_db(self, dbp='test.db'):
        sqlite_connection = None
        print(os.path.exists(dbp))
        if os.path.exists(dbp):
            try:
                sqlite_connection = sqlite3.connect('test.db')
                cursor = sqlite_connection.cursor()
                print("Successfully Connected to SQLite")

                sqlite_select_Query = "select sqlite_version();"
                cursor.execute(sqlite_select_Query)
                record = cursor.fetchall()
                print("SQLite Database Version is: ", record)
                cursor.close()
            except sqlite3.Error as error:
                print("Error while connecting to sqlite", error)
        else:
            try:
                sqlite_connection = sqlite3.connect('test.db')
                cursor = sqlite_connection.cursor()
                print("Database created and Successfully Connected to SQLite")

                sqlite_select_query = "select sqlite_version();"
                cursor.execute(sqlite_select_query)
                record = cursor.fetchall()
                print("SQLite Database Version is: ", record)

                create_user_table = "CREATE TABLE users(id integer PRIMARY KEY, name varchar(255) NOT NULL, embedding text NOT NULL)"
                cursor.execute(create_user_table)
                sqlite_connection.commit()
                print("Default user table created")
                cursor.close()
            except sqlite3.Error as error:
                print("Error while connecting to sqlite", error)
        return sqlite_connection

    def perform_db_optimization(self, cluster_num = 5, min_split_sample = 100):
        try:
            c = self.db.cursor()
            get_all_query = "select embedding from users"
            c.execute(get_all_query)
            record = c.fetchall()
            if len(record) < min_split_sample:
                print(f"Warning: Not enough face stored, min required: {min_split_sample}")
            else:
                record = [ast.literal_eval(x) for x in record]
                kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
                centers = kmeans.cluster_centers_
                with open('centroids.pkl', "wb") as f:
                    pickle.dump(centers, f)

        except sqlite3.Error as error:
            print("Error while fetching user table", error)

    def detect_face(self, img):
        # check if image is path or cv2 image
        if isinstance(img, str):
            img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        boxes, labels, probs = det_predictor.predict(img, 1500 / 2, 0.7)
        assert boxes.size(0) != 1, "multiple faces detected, please retake"
        return boxes

    def generate_embeding(self, cropped):
        input_dataset = ImageData([cropped])
        images_loader = torch.utils.data.DataLoader(
            input_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0,
            drop_last=False)

        for data in images_loader:
            res = []
            t0 = time.time()
            for d in data:
                res.append(normal_recog_net(d).data.cpu().numpy())
            features = np.concatenate((res[0], res[1]), 1)
            features = features / np.expand_dims(np.sqrt(np.sum(np.power(features, 2), 1)), 1)
        return features

    def register(self, name, img):
        if self.db is None:
            raise Exception('Error: db not connected')
        else:
            boxes = self.detect_face(img)
            box = boxes[0,:]
            x1, y1, x2, y2 = pos_box(box)
            cropped = img[y1:y2, x1:x2]
            embedding = self.generate_embeding(cropped)
            embedding_str = str(embedding)
            try:
                c = self.db.cursor()
                insert_query = f"INSERT INTO users (name, embedding) VALUES ({name}, {embedding_str})"
                c.execute(insert_query)
                self.db.commit()
            except:
                self.db.rollback()
                raise Exception('Error: insertion into db failed')

    def delete(self):
        pass
    def recognize(self):
        pass
    def verify(self):
        pass


# class KYCDeepFace:
#     def __init__(self, verify_confidence=0.76, max_image_size=(500, 500), min_image_size=(100, 100), model=None):
#         return

#     def get_faces (self, image_binary_data):
#         return {
#             "code": "SUCCESS",
#             "faces": [
#                 {
#                     "box": { "x1": 0, "x2": 290, "y1": 269, "y2": 488 },
#                     "features": "ABCD=" # The embedding as base64 encoded binary data
#                 },
#                 {
#                     "box": { "x1": 0, "x2": 290, "y1": 269, "y2": 488 },
#                     "features": "ABCD=" # Any other face in the image
#                 }
#             ],
#             "partial_faces": [
#                 # For parts of the image that may almost be detected as face but eventually were deemed non-viable
#                 # The existing errors are depending on what we can detect. Some examples that could work below:
#                 {
#                     "reason": "CROPPED", # Exemplary (unsure if possible): if a face was cropped on top
#                     "box": { "x1": 0, "x2": 0, "y1": 25, "y2": 70 }
#                 },
#                 {
#                     "reason": "NOT_FRONT_FACING", # Face not facing
#                     "box": { "x1": 0, "x2": 0, "y1": 25, "y2": 70 }
#                 },
#                 {
#                     "reason": "MASKED", # Face with a mask
#                     "box": { "x1": 0, "x2": 0, "y1": 25, "y2": 70 }
#                 }
#             ]
#         }
#         return {
#             "code": "INVALID_IMAGE",
#             "reason": "TOO_SMALL" # TOO_LARGE, INVALID_FORMAT, READ_ERROR, TOO_DARK...
#         }
#         return {
#             "code": "ERROR",
#             "message": "Something strange happened"
#         }

#     def match_faces (self, face_1_features, face_2_features):
#         return {
#             "code": "SUCCESS",
#             "confidence": 0.78,
#             "verified": True
#         }
#         return {
#             # Any unexpected error occurred
#             "code": "ERROR",
#             "message": "Something strange happened"
#         }
