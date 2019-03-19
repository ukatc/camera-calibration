import camera_calibration as calib
import cv2 as cv
import numpy as np
from pytest import approx


def test_points_transform_to_expected_mm_values():
    params = cv.SimpleBlobDetector_Params()
    params.minArea = 50
    params.maxArea = 1000
    params.filterByArea = True
    params.minCircularity = 0.2
    params.filterByCircularity = True
    params.blobColor = 0
    params.filterByColor = True
    detector = cv.SimpleBlobDetector_create(params)

    finished, config = calib.Config.generate(
        'mocked checkboard.png', 10, 15,
        'cleaned_grids/distcor_01.bmp', detector, 116, 170, 85, 58
    )

    assert finished, 'Cannot create complete camera calibration config'

    points = np.array([[[3584.902, 2468.0232]],
                       [[71.22837, 2466.539]],
                       [[68.2684, 62.64333]],
                       [[3600.0374, 78.32093]]], np.float32)

    expectations = np.array([[85, 58],
                             [0, 58],
                             [0, 0],
                             [85, 0]], np.float32)

    corrected_points = calib.correct_points(points, config)

    assert points.shape == corrected_points.shape

    for i in range(len(corrected_points)):
        original = points[i, 0]
        point = corrected_points[i, 0]
        expectation = expectations[i]
        assert len(point) == 2

        assert point == approx(expectation, abs=1e-2),\
            'point {} corrects to {}. Expected values within 0.01mm of {}'.format(original, point, expectation)
