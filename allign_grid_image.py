""""
Corrects an image for lens and keystone distortions, and calculates the positions of image points on the reconstructed 2D plane
"""
import numpy as np
import cv2 as cv
import math

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

# Read in a dot pattern image
img = cv.imread('cleaned_grids/distcor_01.bmp')

# Clear the lens distortion
de_lensed = cv.undistort(img, old_camera_matrix, distortion_coefficients, None, new_camera_matrix)

# Locate the corners of the grid


def locate_dot_grid(image, rows, columns):
    params = cv.SimpleBlobDetector_Params()
    params.minArea = 50
    params.maxArea = 1000
    params.filterByArea = True
    params.minCircularity = 0.2
    params.filterByCircularity = True
    params.blobColor = 0
    params.filterByColor = True
    detector = cv.SimpleBlobDetector_create(params)
    gridparams = cv.CirclesGridFinderParameters()
    found, grid = cv.findCirclesGrid(image, (columns, rows),
                                     cv.CALIB_CB_SYMMETRIC_GRID + cv.CALIB_CB_CLUSTERING, detector, gridparams)
    return found, grid


grid_cols, grid_rows = (170, 116)
found, grid = locate_dot_grid(de_lensed, grid_rows, grid_cols)

bottom_right = grid[0, 0]
bottom_left = grid[grid_cols-1, 0]
top_left = grid[-1, 0]
top_right = grid[-grid_cols, 0]

corners = np.array([bottom_right, bottom_left, top_left, top_right])
print('grid corners:\n' + str(corners))

# Correct any keystone distortion


def correct_keystone(image, corners, target_corners, target_img_width, target_img_height):
    """ Corners should be in the same order as the target corners """
    homography_matrix, _ = cv.findHomography(corners, target_corners)
    print('homography matrix:\n' + str(homography_matrix))
    return cv.warpPerspective(image, homography_matrix, (target_img_width, target_img_height))


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


border = 50
grid_width = max(distance(top_left, top_right),
                 distance(bottom_left, bottom_right))
grid_height = max(distance(top_left, bottom_left),
                  distance(top_right, bottom_right))
point_spacing = math.ceil(max(grid_width / (grid_cols - 1),
                              grid_height / (grid_rows - 1)))
grid_width = point_spacing * (grid_cols - 1)
grid_height = point_spacing * (grid_rows - 1)


target_corners = np.array([
    [grid_width + border, grid_height + border],  # bottom right
    [border, grid_height + border],  # bottom left
    [border, border],  # top left
    [grid_width + border, border]  # top right
])
print('target corners:\n' + str(target_corners))
keyed = correct_keystone(de_lensed, corners, target_corners, grid_width + 2 * border, grid_height + 2 * border)

cv.imwrite('keystone_adjusted.png', keyed)

# Convert image points to real world coordinates


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


image_corner_points = Corners(target_corners[2], target_corners[3], target_corners[1], target_corners[0])
space_corner_points = Corners((0, 0), (85, 0), (0, 58), (85, 58))


def test_point_to_real(image_point, image_corners, real_corners, expectation):
    real_point = point_to_real(image_point, image_corners, real_corners)
    print('{} converts to {}. Expected {}'.format(image_point, real_point, expectation))


# Test conversions at the corners
test_point_to_real(image_corner_points.bottom_left, image_corner_points, space_corner_points, space_corner_points.bottom_left)
test_point_to_real(image_corner_points.bottom_right, image_corner_points, space_corner_points, space_corner_points.bottom_right)
test_point_to_real(image_corner_points.top_left, image_corner_points, space_corner_points, space_corner_points.top_left)
test_point_to_real(image_corner_points.top_right, image_corner_points, space_corner_points, space_corner_points.top_right)
# Test a point two dots (1mm) inside the grid
image_point = [image_corner_points.top_left[0] + 2 * point_spacing, image_corner_points.top_left[1] + 2 * point_spacing]
real_point = [space_corner_points.top_left[0] + 1, space_corner_points.top_left[1] + 1]
test_point_to_real(image_point, image_corner_points, space_corner_points, real_point)
