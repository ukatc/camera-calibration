import camera_calibration as calib
import cv2 as cv
import math
import numpy as np
import pytest


def assess_points_transform_to_given_absolute_accuracy(
        config: calib.Config, points: np.ndarray, expectations: np.ndarray, accuracy: float):
    """
    Apply a camera correction to points, testing if the result is within the given accuracy of their expected corrected
    values
    :param accuracy: A point will be accepted if its x and y components are this close to their expected values
    """
    np.set_printoptions(suppress=True)

    corrected_points = calib.correct_points(points, config)

    assert points.shape == corrected_points.shape

    distances = []
    for i in range(len(corrected_points)):
        point = corrected_points[i, 0]
        expectation = expectations[i]
        assert len(point) == 2
        distances.append(math.hypot(point[0] - expectation[0], point[1] - expectation[1]))

    print('image points:')
    print(points)
    print('expected locations:')
    print(expectations)
    print('calculated locations:')
    print(np.array([point[0] for point in corrected_points]))
    print('deviations:')
    print(distances)

    assert max(distances) <= accuracy, 'Expected and calculated points differed by more than the permitted accuracy'


def test_points_transform_from_combined_config_to_hundredth_mm():
    params = cv.SimpleBlobDetector_Params()
    params.minArea = 50
    params.maxArea = 1000
    params.filterByArea = True
    params.minCircularity = 0.2
    params.filterByCircularity = True
    params.blobColor = 0
    params.filterByColor = True
    detector = cv.SimpleBlobDetector_create(params)

    config = calib.Config()

    assert config.populate_distortion_from_chessboard('002h.bmp', 6, 8), 'Unable to populate distortion parameters'
    assert config.populate_homography_from_symmetric_dot_pattern(
        'cleaned_grids/distcor_01.bmp', detector, 116, 170, 84.5, 57.5), 'Unable to populate homography parameters'

    # Points determined by dot centers from running an openCV blob detector over cleaned_grids/distcor_01.bmp
    points = np.array([[[3584.902, 2468.0232]],  # bottom right
                       [[71.22837, 2466.539]],  # bottom left
                       [[68.2684, 62.64333]],  # top left
                       [[3600.0374, 78.32093]],  # top right

                       [[1804.8428, 38.65753]],  # middle top
                       [[1799.092, 2498.543]],  # middle bottom
                       [[47.950756, 1299.2955]],  # middle left
                       [[3611.6602, 1307.9681]],  # middle right
                       [[1800.4049, 1304.3975]],  # center
                       ], np.float32)

    expectations = np.array([[84.5, 57.5],
                             [0, 57.5],
                             [0, 0],
                             [84.5, 0],

                             [83/2, 0],
                             [83/2, 57.5],
                             [0, 59/2],
                             [84.5, 59/2],
                             [83/2, 59/2],
                             ], np.float32)

    assess_points_transform_to_given_absolute_accuracy(config, points, expectations, 0.005)
