from __future__ import print_function
import camera_calibration as calib
import cv2 as cv
import math
import numpy as np
import pprint
import pytest
import sys


def assess_points_transform_to_given_absolute_accuracy(
    config, points, expectations, accuracy
):
    # type: (calib.Config, np.ndarray, np.ndarray, float) -> None
    """
    Apply a camera correction to points, testing if the result is within the given accuracy of their expected corrected
    values
    :param config: The calibration configuration for a camera to test
    :param points: An (N x 1 x 2) numpy array of points in the camera's image space to correct
    :param expectations: An (N x 2) numpy array of points in real space that the points array is expected to correct to
    :param accuracy: To pass, the config should transform all input points to within this distance of their expectation
    """
    np.set_printoptions(suppress=True)

    corrected_points = calib.correct_points(
        points, config, calib.Correction.lens_keystone_and_real_coordinates
    )

    assert points.shape == corrected_points.shape

    distances = []
    distance_hist = {}
    highest = 0
    for i in range(len(corrected_points)):
        original = points[i, 0]
        point = corrected_points[i, 0]
        expectation = expectations[i]
        assert len(point) == 2
        distance = math.hypot(point[0] - expectation[0], point[1] - expectation[1])
        if distance > highest:
            highest = distance
            print(
                "new highest: px{} res{} exp{} dist{}".format(
                    original, point, expectation, distance
                )
            )
        distances.append(distance)
        microns = math.ceil(distance * 1000)
        if microns in distance_hist:
            distance_hist[microns] += 1
        else:
            distance_hist[microns] = 1

    print("average deviation:\n{}mm".format(sum(distances) / len(distances)))
    print("max deviation:\n{}mm".format(max(distances)))
    print("deviation spread:")
    pprint.pprint(distance_hist)

    assert (
        max(distances) <= accuracy
    ), "Expected and calculated points differed by more than the permitted accuracy"


@pytest.mark.slow
def test_points_transform_from_combined_config_to_100_microns():
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

    assert config.populate_lens_parameters_from_chessboard(
        "sample_images/002h.bmp", 6, 8
    ), "Unable to populate distortion parameters"
    assert config.populate_keystone_and_real_parameters_from_symmetric_dot_pattern(
        "sample_images/distcor_01_cleaned.bmp", detector, 116, 170, 84.5, 57.5
    ), "Unable to populate homography parameters"

    # Points determined by dot centers from running an openCV blob detector over sample_images/distcor_01_cleaned.bmp
    points = np.array(
        [
            [[3584.902, 2468.0232]],  # bottom right
            [[71.22837, 2466.539]],  # bottom left
            [[68.2684, 62.64333]],  # top left
            [[3600.0374, 78.32093]],  # top right
            [[1804.8428, 38.65753]],  # middle top
            [[1799.092, 2498.543]],  # middle bottom
            [[47.950756, 1299.2955]],  # middle left
            [[3611.6602, 1307.9681]],  # middle right
            [[1800.4049, 1304.3975]],  # center
        ],
        np.float32,
    )

    expectations = np.array(
        [
            [84.5, 57.5],
            [0, 57.5],
            [0, 0],
            [84.5, 0],
            [83 / 2.0, 0],
            [83 / 2.0, 57.5],
            [0, 59 / 2.0],
            [84.5, 59 / 2.0],
            [83 / 2.0, 59 / 2.0],
        ],
        np.float32,
    )

    assess_points_transform_to_given_absolute_accuracy(
        config, points, expectations, 0.1
    )


@pytest.mark.slow
def test_points_transform_from_only_dot_grid_to_20_microns():
    config = calib.Config()
    assert config.populate_lens_parameters_from_chessboard(
        "sample_images/002h.bmp", 6, 8
    ), "Unable to populate distortion parameters"

    params = cv.SimpleBlobDetector_Params()
    params.minArea = 50
    params.maxArea = 1000
    params.filterByArea = True
    params.minCircularity = 0.2
    params.filterByCircularity = True
    params.blobColor = 0
    params.filterByColor = True
    dot_detector = cv.SimpleBlobDetector_create(params)

    rows = 116
    cols = 170

    dot_image = cv.imread("sample_images/distcor_01_cleaned.bmp")
    undistorted_dot_image = calib.correct_image(
        dot_image, config, calib.Correction.lens_distortion
    )
    print("searching for grid in undistorted image")
    found, undistorted_grid = cv.findCirclesGrid(
        undistorted_dot_image,
        (cols, rows),
        cv.CALIB_CB_SYMMETRIC_GRID + cv.CALIB_CB_CLUSTERING,
        dot_detector,
        cv.CirclesGridFinderParameters(),
    )
    assert found, "Unable to find dot grid in initially undistorted image"
    print("grid found in undistorted image")

    print("looking for dots in distorted image")
    distorted_points = cv.KeyPoint_convert(dot_detector.detect(dot_image))
    print("dots found in distorted image")
    distorted_points = np.array([[point] for point in distorted_points], np.float32)
    transformed_points = cv.undistortPoints(
        distorted_points,
        config.distorted_camera_matrix,
        config.distortion_coefficients,
        P=config.undistorted_camera_matrix,
    )

    def distance(p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    print("searching for dot mapping")
    distorted_grid = np.zeros(undistorted_grid.shape, undistorted_grid.dtype)
    for i in range(len(undistorted_grid)):
        if i % 100 == 0:
            print("progress: {}/{}".format(i, cols * rows), end="\r")
        # get the point at i in the grid
        grid_member = undistorted_grid[i]
        # find the nearest member of transformed_points
        nearest_distance = sys.float_info.max
        original_point = None
        for j in range(len(transformed_points)):
            transformed_point = transformed_points[j]
            separation = distance(grid_member[0], transformed_point[0])
            if separation < nearest_distance:
                nearest_distance = separation
                original_point = distorted_points[j]
            if separation < 1:
                break
        # get the untransformed point that matches the transformed_point
        assert original_point is not None
        # set it to position i in the new grid
        distorted_grid[i, 0, 0] = original_point[0, 0]
        distorted_grid[i, 0, 1] = original_point[0, 1]

    # np.save('full_grid', distorted_grid)

    print("generating sparse grid")
    # Use fewer points to improve distortion performance from not finishing in >20mins while using all available RAM
    sparse_rows = rows // 2
    sparse_cols = cols // 2
    sparse_grid = np.zeros((sparse_rows * sparse_cols, 1, 2), np.float32)
    for i in range(sparse_rows):
        for j in range(sparse_cols):
            sparse_grid[i * sparse_cols + j, 0, 0] = distorted_grid[
                (i * 2) * cols + (j * 2), 0, 0
            ]
            sparse_grid[i * sparse_cols + j, 0, 1] = distorted_grid[
                (i * 2) * cols + (j * 2), 0, 1
            ]

    print("generating config")
    h, w = dot_image.shape[:2]
    dot_config = calib.Config()
    dot_config.populate_lens_parameters_from_grid(
        sparse_grid, sparse_cols, sparse_rows, w, h
    )

    undistorted_distorted_grid = cv.undistortPoints(
        distorted_grid,
        dot_config.distorted_camera_matrix,
        dot_config.distortion_coefficients,
        P=dot_config.undistorted_camera_matrix,
    )

    dot_config.populate_keystone_and_real_parameters_from_grid(
        undistorted_distorted_grid, cols, rows, 84.5, 57.5
    )

    print("initial config:")
    print(config)
    print("dot config:")
    print(dot_config)

    expectations = np.zeros((len(undistorted_grid), 2), np.float32)
    for i in range(rows):
        for j in range(cols):
            expectations[i * cols + j, 0] = 84.5 - (0.5 * j)
            expectations[i * cols + j, 1] = 57.5 - (0.5 * i)

    assess_points_transform_to_given_absolute_accuracy(
        dot_config, distorted_grid, expectations, 0.02
    )


def test_points_transform_from_only_chessboard_to_100_microns():
    config = calib.Config()
    assert config.populate_lens_parameters_from_chessboard(
        "sample_images/002h.bmp", 6, 8
    ), "Unable to populate distortion parameters"
    assert config.populate_keystone_and_real_parameters_from_chessboard(
        "sample_images/002h.bmp", 8, 6, 90.06, 64.45
    ), "Unable to populate homography parameters"

    found, corners = cv.findChessboardCorners(
        cv.imread("sample_images/002h.bmp"), (8, 6)
    )
    targets = np.zeros((len(corners), 2), np.float32)
    for i in range(8):
        for j in range(6):
            targets[j * 8 + i, 0] = 90.06 * (7 - i) / 7.0
            targets[j * 8 + i, 1] = 64.45 * (5 - j) / 5.0

    assess_points_transform_to_given_absolute_accuracy(config, corners, targets, 0.1)


def test_points_transform_from_only_mock_chessboard_to_100_microns():
    config = calib.Config()
    assert config.populate_lens_parameters_from_chessboard(
        "sample_images/mocked checkboard.png", 10, 15
    ), "Unable to populate distortion parameters"
    assert config.populate_keystone_and_real_parameters_from_chessboard(
        "sample_images/mocked checkboard.png", 15, 10, 70, 45
    ), "Unable to populate homography parameters"

    found, corners = cv.findChessboardCorners(
        cv.imread("sample_images/mocked checkboard.png"), (15, 10)
    )
    targets = np.zeros((len(corners), 2), np.float32)
    for i in range(15):
        for j in range(10):
            targets[j * 15 + i, 0] = 70 * (14 - i) / 14.0
            targets[j * 15 + i, 1] = 45 * (9 - j) / 9.0

    assess_points_transform_to_given_absolute_accuracy(config, corners, targets, 0.1)


def test_homography_border_doesnt_affect_point_transforms():
    default_border_config = calib.Config()
    assert default_border_config.populate_lens_parameters_from_chessboard(
        "sample_images/002h.bmp", 6, 8
    ), "Unable to populate distortion parameters"
    assert default_border_config.populate_keystone_and_real_parameters_from_chessboard(
        "sample_images/002h.bmp", 8, 6, 90.06, 64.45
    ), "Unable to populate homography parameters"

    custom_border_config = calib.Config()
    assert custom_border_config.populate_lens_parameters_from_chessboard(
        "sample_images/002h.bmp", 6, 8
    ), "Unable to populate distortion parameters"
    assert custom_border_config.populate_keystone_and_real_parameters_from_chessboard(
        "sample_images/002h.bmp", 8, 6, 90.06, 64.45, border=200
    ), "Unable to populate homography parameters"

    found, corners = cv.findChessboardCorners(
        cv.imread("sample_images/mocked checkboard.png"), (15, 10)
    )
    targets = np.zeros((len(corners), 2), np.float32)
    for i in range(15):
        for j in range(10):
            targets[j * 15 + i, 0] = 70 * (14 - i) / 14.0
            targets[j * 15 + i, 1] = 45 * (9 - j) / 9.0

    default_border_transformed = calib.correct_points(
        corners,
        default_border_config,
        calib.Correction.lens_keystone_and_real_coordinates,
    )
    custom_border_transformed = calib.correct_points(
        corners,
        custom_border_config,
        calib.Correction.lens_keystone_and_real_coordinates,
    )

    assert np.allclose(
        default_border_transformed, custom_border_transformed
    ), "Real world coordinates not consistent with image borders"
