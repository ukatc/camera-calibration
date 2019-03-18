import numpy as np
import cv2 as cv

# Camera calibration parameters determined by checkers.py
old_camera_matrix = np.array([[8.99155523e+03, 0.00000000e+00, 1.96472471e+03],
                              [0.00000000e+00, 8.98921557e+03, 1.34033122e+03],
                              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
rotate_vector = np.array([[-0.0068679],
                          [-0.03979584],
                          [1.5731318]])
translate_vector = np.array([[34.7388702],
                             [-27.7625536],
                             [207.90678648]])
distortion_coefficients = np.array([
    [-7.36345426e-01,
     2.66415181e+00,
     -4.23351460e-03,
     -2.60186137e-03,
     -1.61495764e+01]])
new_camera_matrix = np.array([[8.54052246e+03, 0.00000000e+00, 1.96340416e+03],
                              [0.00000000e+00, 8.55686035e+03, 1.33381448e+03],
                              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

points = np.array([[[3584.902, 2468.0232]]], np.float32)

undistorted = cv.undistortPoints(points, old_camera_matrix, distortion_coefficients, P=new_camera_matrix)
print(undistorted)

# Homography matrix using the distcor_01 grid, and the above distortion correction, determined by allign_grid_image.py
homography_matrix = np.array([[1.00389426e+00,  -1.82376659e-03, -3.53075992e+01],
                              [-6.70797235e-03,  1.00522456e+00, -1.69727256e+01],
                              [-3.41044536e-06, -2.41122605e-06,  1.00000000e+00]])

flattened = cv.perspectiveTransform(undistorted, homography_matrix)
print(flattened)


class Corners:
    def __init__(self, top_left, top_right, bottom_left, bottom_right):
        self.top_left = top_left
        self.top_right = top_right
        self.bottom_left = bottom_left
        self.bottom_right = bottom_right


def point_to_real(image_point, image_corners: Corners, real_corners: Corners):
    real_x_range = real_corners.top_right[0] - real_corners.top_left[0]
    real_y_range = real_corners.bottom_left[1] - real_corners.top_left[1]

    pixel_x_range = image_corners.top_right[0] - image_corners.top_left[0]
    pixel_y_range = image_corners.bottom_left[1] - image_corners.top_left[1]

    x_fraction = (image_point[0] - image_corners.top_left[0]) / pixel_x_range
    y_fraction = (image_point[1] - image_corners.top_left[1]) / pixel_y_range

    x = real_corners.top_left[0] + x_fraction * real_x_range
    y = real_corners.top_left[1] + y_fraction * real_y_range

    return x, y


image_corner_points = Corners((50, 50), (3599, 50), (50, 2465), (3599, 2465))
space_corner_points = Corners((0, 0), (85, 0), (0, 58), (85, 58))

location = point_to_real(flattened[0, 0], image_corner_points, space_corner_points)
print(location)
