import numpy as np
import cv2 as cv

np.set_printoptions(suppress=True)

# Camera calibration parameters determined by checkers.py
old_camera_matrix = np.array([[8.99155523e+03, 0.00000000e+00, 1.96472471e+03],
                              [0.00000000e+00, 8.98921557e+03, 1.34033122e+03],
                              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
distortion_coefficients = np.array([
    [-7.36345426e-01,
     2.66415181e+00,
     -4.23351460e-03,
     -2.60186137e-03,
     -1.61495764e+01]])
new_camera_matrix = np.array([[8.54052246e+03, 0.00000000e+00, 1.96340416e+03],
                              [0.00000000e+00, 8.55686035e+03, 1.33381448e+03],
                              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

points = np.array([[[3584.902, 2468.0232]],
                   [[71.22837, 2466.539]],
                   [[68.2684, 62.64333]],
                   [[3600.0374, 78.32093]]], np.float32)

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


def points_to_real(image_points: np.ndarray, image_corners: Corners, real_corners: Corners):
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


image_corner_points = Corners((50, 50), (3599, 50), (50, 2465), (3599, 2465))
space_corner_points = Corners((0, 0), (85, 0), (0, 58), (85, 58))

locations = points_to_real(flattened, image_corner_points, space_corner_points)
print(locations)
