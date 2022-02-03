#!/usr/bin/env python
from unittest import main, TestSuite, makeSuite
from core.face_landmarks_test import FaceLandmarkTest
from vision.ssd.config.fd_config_test import ImageConfigurationTest
from vision.utils.geom_test import GeomTest
from vision.utils.lang_test import LazyTest, LangTest
from vision.utils.image_loader_test import ImageLoaderTest
from vision.landmark_detector_test import LandmarkDetectorTest

suite = TestSuite()
suite.addTests(map(makeSuite, [
    ImageConfigurationTest,
    LazyTest,
    LangTest,
    GeomTest,
    FaceLandmarkTest,
    ImageLoaderTest,
    LandmarkDetectorTest,
]))

if __name__ == '__main__':
    main()
