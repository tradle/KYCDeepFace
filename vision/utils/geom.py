import numpy as np
from vision.utils.lang import lazy

class Line:
    def __init__(self, point_a: tuple, point_b: tuple):
        self.point_a = point_a
        self.point_b = point_b
        x = point_b[0] - point_a[0]
        y = point_b[1] - point_a[1]

        self._distance = lazy(lambda: np.sqrt((x ** 2) + (y ** 2)))
        self._angle = lazy(lambda: np.degrees(np.arctan2(y, x)))
        self._center = lazy(lambda: (point_a[0] + x / 2, point_a[1] + y / 2))

    def __len__(self):
        return 2

    def __iter__(self):
        return iter((self.point_a, self.point_b))

    @property
    def distance(self) -> float:
        return self._distance()
    
    @property
    def angle(self) -> float:
        return self._angle()
    
    @property
    def center(self):
        return self._center()
