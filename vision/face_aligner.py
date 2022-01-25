import cv2
from core.face_landmarks import FaceLandmarks

from vision.utils.geom import Line

class FaceAligner:
    def __init__(self, left_eye: tuple=(0.30, 0.30), dsize: tuple=(256.0, 256.0)):
        self.dsize = dsize
        w, h = dsize
        x = left_eye[0] * w
        y = left_eye[1] * h
        self.desired_eye_line = Line([x, y], [w - x, y])

    def align_face(self, image, face_landmarks: FaceLandmarks):
        return self.align_eyes(image, face_landmarks.eye_line)

    def align_eyes(self, image, line_between_eyes: Line):
        center = line_between_eyes.center
        target_center = self.desired_eye_line.center
        scale = self.desired_eye_line.distance / line_between_eyes.distance

        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(center, line_between_eyes.angle, scale)
        # translate the matrix to move the eyes up
        M[0, 2] += (target_center[0] - center[0])
        M[1, 2] += (target_center[1] - center[1])

        # apply the affine transformation
        return cv2.warpAffine(image, M, self.dsize, flags=cv2.INTER_CUBIC)
