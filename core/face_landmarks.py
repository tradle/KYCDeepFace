# -*-coding:utf-8-*-
import cv2
import numpy as np
from vision.utils.geom import Line
from vision.utils.lang import lazy, float_rounded, format_point, format_points

# TODO: Where do these coordinates come from? (what kind of object is this? a face?)
# see headpose.html - probably the eye parts of the face?
object_pts = np.float32([
    [ 6.825897, 6.760612, 4.402142],
    [ 1.330353, 7.122144, 6.903745],
    [-1.330353, 7.122144, 6.903745],
    [-6.825897, 6.760612, 4.402142],
    [ 5.311432, 5.485328, 3.987654],
    [ 1.789930, 5.393625, 4.413414],
    [-1.789930, 5.393625, 4.413414],
    [-5.311432, 5.485328, 3.987654],
    [ 2.005628, 1.409845, 6.165652],
    [-2.005628, 1.409845, 6.165652],
])

def box_3d(width: float, height: float=None, depth: float=None, left: float=None, top: float=None, front: float=None):
    # cube if lengths are undefined
    if height is None: height = width
    if depth is None: depth = width

    # center if t/l/f is undefined
    if left is None: left = -(width / 2)
    if top is None: top = -(height / 2)
    if front is None: front = -(depth / 2)
    right = left + width
    bottom = top + height
    back = front + depth
    return np.float32([
        [right, bottom, back],
        [right, bottom, front],
        [right, top, front],
        [right, top, back],
        [left, bottom, back],
        [left, bottom, front],
        [left, top, front],
        [left, top, back],
    ])

# TODO: why is the reprojection cube of size=20?
reprojectsrc = box_3d(20)

class PointGroup:
    def __init__(self, points):
        self.points = points

    def __len__(self):
        return len(self.points)

    def __iter__(self):
        return iter(self.points)

    @property
    def center(self):
        return self.points.mean(axis=0).astype("float")

    @property
    def middle_pt(self):
        return self.points[len(self.points) >> 1]

    @property
    def first_pt(self):
        return self.points[0]
    
    @property
    def last_pt(self):
        return self.points[-1]

class FaceLandmarks(PointGroup):
    def __init__(
        self,
        points: list,
        shape: list,
        bbox: list
    ):
        super().__init__(points)
        self.shape = shape
        self.bbox = bbox
        self.outline = PointGroup(points[0:17])
        self.left_brows = PointGroup(points[17:22])
        self.right_brows = PointGroup(points[22:27])
        self.nose_back = PointGroup(points[27:31])
        self.nostrils = PointGroup(points[31:36])
        self.left_eye = PointGroup(points[36:42])
        self.right_eye = PointGroup(points[42:48])
        self.mouth = PointGroup(points[48:68])
        self._eye_line = lazy(lambda: Line(self.left_eye.center, self.right_eye.center))
        self._head_pose = lazy(lambda: self.__head_pose())
    
    @property
    def eye_line(self) -> Line:
        return self._eye_line()

    @property
    def angle(self):
        _, angle = self.head_pose
        return angle

    @property
    def head_pose(self):
        return self._head_pose()

    def __head_pose(self):
        w, h = self.shape
        cam_matrix = np.float32([
            w, 0.0, w // 2,
            0.0, w, h // 2,
            0.0, 0.0, 1.0
        ]).reshape(3, 3)

        dist_coeffs = np.float32(
            [0, 0, 0, 0, 0]
        ).reshape(5, 1)

        image_pts = np.float32([
            self.left_brows.first_pt, self.left_brows.last_pt,
            self.right_brows.first_pt, self.right_brows.last_pt,
            self.left_eye.first_pt, self.left_eye.middle_pt,
            self.right_eye.first_pt, self.right_eye.middle_pt,
            self.nostrils.first_pt, self.nostrils.last_pt
        ])
        _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)
        reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix, dist_coeffs)
        reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pose_mat = cv2.hconcat((rotation_mat, translation_vec))
        _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)
        pitch, yaw, roll = euler_angle[:, 0]
        # TODO: how do we use reprojectdst? Is it of any value?
        return reprojectdst, [pitch, yaw, roll]

def format_landmarks(landmarks: FaceLandmarks):
    box = landmarks.bbox
    pitch, yaw, roll = landmarks.angle
    return {
        'bounds': {
            'topLeft': format_point(box[0], box[1]),
            'bottomRight': format_point(box[2], box[3])
        },
        'landmarks': {
            'outline': list(format_points(landmarks.outline)),
            'left_brows': list(format_points(landmarks.left_brows)),
            'right_brows': list(format_points(landmarks.right_brows)),
            'nose_back': list(format_points(landmarks.nose_back)),
            'nostrils': list(format_points(landmarks.nostrils)),
            'left_eye': list(format_points(landmarks.left_eye)),
            'right_eye': list(format_points(landmarks.right_eye)),
            'mouth': list(format_points(landmarks.mouth)),
        },
        'angles': {
            'pitch': float_rounded(pitch, 3),
            'yaw': float_rounded(yaw, 3),
            'roll': float_rounded(roll, 3)
        }
    }
