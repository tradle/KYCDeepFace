#!/usr/bin/env python3 -m bin.webcam-alignment-example
# -*- coding: utf-8 -*-
# @Author: yirui
# @Date:   2021-09-01 17:39:28
# @Last Modified by:   yirui
# @Last Modified time: 2021-09-01 17:40:40
if __name__ != '__main__':
    raise Exception('This is supposed to be run as a cli script.')

import cv2

from config import embedding_setup
from vision.kycdeepface import face_embeddings, face_match
from vision.utils.lang import format_timings_str
import random

def clean_inactive(faces):
    inactive = 0
    for face in faces:
        if face.active:
            continue
        cv2.destroyWindow(face.id)
        if inactive >= 10:
            faces.remove(face)
        else:
            inactive += 1

    return faces

global_face_index = 0
class FaceState:
    def __init__(self, face):
        global global_face_index

        global_face_index += 1
        _, __, embedding = face

        self.active = True
        self.embedding = embedding
        self.id = f'face#{global_face_index} ({random.randbytes(6).hex()})'

def activate_face(face, faces, threshold=0.5):
    _, __, embeddings = face

    face_state = None
    best_match = 0.0
    for target_face in faces:
        if target_face.active:
            continue
        
        match = face_match(embeddings, target_face.embedding)

        if best_match < match:
            best_match = match
            face_state = target_face
    
    print(f'best_match={best_match}')
    if best_match < threshold:
        face_state = FaceState(face)
        faces.append(face_state)

    # To match similarity to previous face
    face_state.embedding = embeddings
    face_state.active = True
    return face_state

def main ():

    faces = []
    cap = cv2.VideoCapture(0)
    while True:
        ret, orig_image = cap.read()

        if orig_image is None:
            print("end")
            break

        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        h, w, _ = orig_image.shape
        dsize = (int(400 / h * w), 400)
        cv2.imshow('input', cv2.resize(orig_image, dsize))
        
        # Deactivate all known faces
        for face_state in faces: face_state.active = False

        timings, found_faces = face_embeddings(embedding_setup, image)
        for found_face in found_faces:
            face_state = activate_face(found_face, faces)
            landmarks, image, embedding = found_face
            cv2.imshow(face_state.id, image)

        clean_inactive(faces)

        print(format_timings_str(timings))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f'closing...')
            break

    cap.release()
    cv2.destroyAllWindows()

main()
