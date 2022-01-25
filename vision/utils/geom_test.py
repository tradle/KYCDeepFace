from unittest.case import TestCase
from vision.utils.geom import Line
import unittest

def test_center(test: TestCase, a, b, expect):
    test.assertTupleEqual(Line(a, b).center, expect, f"center({a}, {b})")
    test.assertTupleEqual(Line(b, a).center, expect, f"center({b}, {a})")

def test_distance(test: TestCase, a, b, expect):
    test.assertEqual(Line(a, b).distance, expect, f"distance({a}, {b})")
    test.assertEqual(Line(b, a).distance, expect, f"distance({b}, {a})")

class GeomTest(unittest.TestCase):
    def test_line_list(self):
        self.assertEqual(len(Line((0, 0), (1, 1))), 2)
        self.assertListEqual(list(Line((0, 0), (1, 1))), [(0, 0), (1, 1)])

    def test_line_center(self):
        test_center(self, (0, 0), (0, 0), (0, 0))
        test_center(self, (0, 0), (0, 1), (0, .5))
        test_center(self, (0, 0), (1, 0), (.5, 0))
        test_center(self, (0, 0), (1, 1), (.5, .5))
        test_center(self, (1, 1), (1, 1), (1, 1))
        test_center(self, (-1, 0), (0, 1), (-.5, .5))
        test_center(self, (1, 1), (2, 2), (1.5, 1.5))

    def test_line_distance(self):
        test_distance(self, (0, 0), (0, 0), 0)
        test_distance(self, (0, 0), (0, 1), 1)
        test_distance(self, (0, 0), (1, 0), 1)
        test_distance(self, (0, 0), (1, 1), 1.4142135623730951)
        test_distance(self, (1, 1), (1, 1), 0)
        test_distance(self, (0.5, 0.5), (0.7, 0.9), 0.4472135954999579)
    
    def test_line_angle(self):
        self.assertEqual(Line((0, 0), (0, 0)).angle, .0)
        self.assertEqual(Line((0, 0), (-1, -.00001)).angle, -179.9994270422049)
        self.assertEqual(Line((0, 0), (-.75, -.5)).angle, -146.30993247402023)
        self.assertEqual(Line((0, 0), (-.5, -.5)).angle, -135.0)
        self.assertEqual(Line((0, 0), (-.25, -.5)).angle, -116.56505117707799)
        self.assertEqual(Line((0, 0), (0, -1)).angle, -90.0)
        self.assertEqual(Line((0, 0), (.25, -.5)).angle, -63.43494882292201)
        self.assertEqual(Line((0, 0), (.5, -.5)).angle, -45.0)
        self.assertEqual(Line((0, 0), (.75, -.25)).angle, -18.43494882292201)
        self.assertEqual(Line((0, 0), (1, 0)).angle, .0)
        self.assertEqual(Line((0, 0), (.75, .25)).angle, 18.43494882292201)
        self.assertEqual(Line((0, 0), (.5, .5)).angle, 45.0)
        self.assertEqual(Line((0, 0), (.25, .5)).angle, 63.43494882292201)
        self.assertEqual(Line((0, 0), (0, 1)).angle, 90.0)
        self.assertEqual(Line((0, 0), (-.25, .5)).angle, 116.56505117707799)
        self.assertEqual(Line((0, 0), (-.5, .5)).angle, 135.0)
        self.assertEqual(Line((0, 0), (-.75, .5)).angle, 146.30993247402023)
        self.assertEqual(Line((0, 0), (-1, 0)).angle, 180.0)
