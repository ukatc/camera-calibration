"""
Functions for correcting distortion in images, and coordinate points within them
"""
import camera_calibration.configuration as conf
import cv2 as cv
from enum import IntEnum
import numpy as np


class Correction(IntEnum):
    """
    Enum to indicate corrections to perform on images and points

    Single transforms have power of two values, so bitwise-and should be used to tell if a single step is required.
    The individual corrections are:
    - lens_distortion - Corrects for the camera's lens distortion
    - keystone_distortion - Corrects for any angle offset from the camera and the calibration image's normal.
    When applied to the calibration image used, the calibration grid will appear rectangular
    - real_coordinates - Only affects points. Project's the pixel coordinates to a position on the
    calibration image's plane based on the grid's corners and size
    """

    lens_distortion = 1
    keystone_distortion = 2
    lens_and_keystone = 3
    real_coordinates = 4
    lens_keystone_and_real_coordinates = 7


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


def correct_points(points, config, correction_level):
    # type: (np.ndarray, conf.Config, Correction) -> np.ndarray
    """
    Map points in pixel space to unit coordinates on the image plane, correcting for lens and keystone distortions
    :param points: Points in the pixel space of a cameras images
    :param config: Image correction configuration
    :param correction_level: The corrections to apply to the points, eg lens_distortion, keystone_distortion,
    real_coordinates, or a combination of those
    :return: An array of the image points mapped to the config's real coordinate system
    """
    if correction_level & Correction.lens_distortion:
        points = cv.undistortPoints(
            points,
            config.distorted_camera_matrix,
            config.distortion_coefficients,
            P=config.undistorted_camera_matrix,
        )
    if correction_level & Correction.keystone_distortion:
        points = cv.perspectiveTransform(points, config.homography_matrix)
    if correction_level & Correction.real_coordinates:
        points = grid_points_to_real(
            points, config.grid_image_corners, config.grid_space_corners
        )
    return points


def correct_point(point, config, correction_level):
    # type: ((float, float), conf.Config, Correction) -> (float, float)
    """
    Map a point in pixel space to unit coordinates on the image plane, correcting for lens and keystone distortions
    :param point: (x, y) point in the pixel space of a cameras images
    :param config: Image correction configuration
    :param correction_level: The corrections to apply to the point, eg lens_distortion, keystone_distortion,
    real_coordinates, or a combination of those
    :return: An (x, y) pair of the image points mapped to the config's real coordinate system
    """
    points = np.array([[[point[0], point[1]]]])
    corrected = correct_points(points, config, correction_level)
    return corrected[0, 0, 0], corrected[0, 0, 1]


def correct_image(img, config, correction_level):
    # type: (np.ndarray, conf.Config, Correction) -> np.ndarray
    """
    Generates a copy of the given image with the chosen distortion correction(s) applied
    :param img: An OpenCV image with camera distortions to correct
    :param config: A calibration configuration for the camera that took this image
    :param correction_level: The correction(s) to apply to the image, eg lens_distortion, keystone_distortion, or
    lens_and_keystone to perform both
    :return: A corrected copy of the image
    """
    if correction_level & Correction.lens_distortion:
        img = cv.undistort(
            img,
            config.distorted_camera_matrix,
            config.distortion_coefficients,
            None,
            config.undistorted_camera_matrix,
        )
    if correction_level & Correction.keystone_distortion:
        target_img_size = (
            int(
                config.grid_image_corners.top_left[0]
                + config.grid_image_corners.bottom_right[0]
            ),
            int(
                config.grid_image_corners.top_left[1]
                + config.grid_image_corners.bottom_right[1]
            ),
        )
        img = cv.warpPerspective(img, config.homography_matrix, target_img_size)
    return img
