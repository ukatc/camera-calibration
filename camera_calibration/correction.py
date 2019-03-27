"""
Functions for correcting distortion in images, and coordinate points within them
"""
import camera_calibration.configuration as conf
import cv2 as cv
import numpy as np


def grid_points_to_real(image_points, image_corners, real_corners):
    # type: (np.ndarray, conf.Corners, conf.Corners) -> np.ndarray
    """
    Map pixel-space coordinates in an image space to a coordinate system based on the given corner mappings
    :param image_points: Array of image space points to be mapped
    :param image_corners: Pixel points that mark the corners of a rectangle in the image
    :param real_corners: Plane coordinates of the rectangle coordinates
    :return: An array of the image points mapped to the given real coordinate system
    """
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


def correct_points(points, config):
    # type: (np.ndarray, conf.Config) -> np.ndarray
    """
    Map points in pixel space to unit coordinates on the image plane, correcting for lens and keystone distortions
    :param points: Points in the pixel space of a cameras images
    :param config: Image correction configuration
    :return: An array of the image points mapped to the config's real coordinate system
    """
    undistorted = cv.undistortPoints(points,
                                     config.distorted_camera_matrix,
                                     config.distortion_coefficients,
                                     P=config.undistorted_camera_matrix)
    flattened = cv.perspectiveTransform(undistorted, config.homography_matrix)
    return grid_points_to_real(flattened, config.grid_image_corners, config.grid_space_corners)


def correct_point(point, config):
    # type: ((float, float), conf.Config) -> (float, float)
    """
    Map a point in pixel space to unit coordinates on the image plane, correcting for lens and keystone distortions
    :param point: (x, y) point in the pixel space of a cameras images
    :param config: Image correction configuration
    :return: An (x, y) pair of the image points mapped to the config's real coordinate system
    """
    points = np.array([[[point[0], point[1]]]])
    corrected = correct_points(points, config)
    return corrected[0, 0, 0], corrected[0, 0, 1]


def correct_camera_distortion(img, config):
    # type: (np.ndarray, conf.Config) -> np.ndarray
    """
    Generates a copy of the given image with lens distortion corrected
    :param img: An openCV image
    :param config: Correction configuration for the camera that took the image
    :return: A copy of the given image with lens distortion corrected
    """
    return cv.undistort(img,
                        config.distorted_camera_matrix,
                        config.distortion_coefficients,
                        None,
                        config.undistorted_camera_matrix)


def correct_keystone_distortion(img, config, target_img_size=None):
    # type: (np.ndarray, conf.Config, (float, float)) -> np.ndarray
    """
    Generates a copy of the given image with keystone distortion corrected
    :param img: An openCV image
    :param config: Correction configuration for the camera that took the image
    :param target_img_size: Optional (width, height) pair for the corrected image
    :return: A copy of the given image with keystone distortion corrected
    """
    if target_img_size is None:
        target_img_size = (
            config.grid_image_corners.top_left[0] + config.grid_image_corners.bottom_right[0],
            config.grid_image_corners.top_left[1] + config.grid_image_corners.bottom_right[1],
        )
    return cv.warpPerspective(img, config.homography_matrix, target_img_size)
