import camera_calibration.configuration as conf
import cv2 as cv
import numpy as np


def grid_points_to_real(image_points: np.ndarray, image_corners: conf.Corners, real_corners: conf.Corners):
    real_points = np.zeros(image_points.shape, np.float32)

    real_x_range = real_corners.top_right[0] - real_corners.top_left[0]
    real_y_range = real_corners.bottom_left[1] - real_corners.top_left[1]

    pixel_x_range = image_corners.top_right[0] - image_corners.top_left[0]
    pixel_y_range = image_corners.bottom_left[1] - image_corners.top_left[1]

    for i in range(len(image_points)):
        image_point = image_points[i, 0]

        x_fraction = (image_point[0] - image_corners.top_left[0]) / pixel_x_range
        y_fraction = (image_point[1] - image_corners.top_left[1]) / pixel_y_range

        x = real_corners.top_left[0] + x_fraction * real_x_range
        y = real_corners.top_left[1] + y_fraction * real_y_range

        real_points[i, 0, 0] = x
        real_points[i, 0, 1] = y

    return real_points


def correct_points(points: np.ndarray, config: conf.Config):
    undistorted = cv.undistortPoints(points,
                                     config.distorted_camera_matrix,
                                     config.distortion_coefficients,
                                     P=config.undistorted_camera_matrix)
    flattened = cv.perspectiveTransform(undistorted, config.homography_matrix)
    return grid_points_to_real(flattened, config.grid_image_corners, config.grid_space_corners)


def correct_point(point, config: conf.Config):
    points = np.array([[[point[0], point[1]]]])
    corrected = correct_points(points, config)
    return corrected[0, 0, 0], corrected[0, 0, 1]
