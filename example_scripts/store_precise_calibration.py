"""
Generate a high accuracy calibration configuration, and save it to file.
The configuration is generated from an image of a high precision grid of dots.
However, for OpenCV to be able to recognise the grid, we need to first correct most of the lens
distortion using a less accurate calibration.
"""
from __future__ import print_function
import camera_calibration as calib
import cv2 as cv
import math
import numpy as np
import sys

calibration_file_name = "calibration.npz"
dot_grid_image_path = "../sample_images/distcor_04_cleaned.bmp"
chessboard_image_path = "../sample_images/002h.bmp"
chess_grid_rows = 6
chess_grid_cols = 8
dot_grid_rows = 116
dot_grid_cols = 170
dot_spacing = 0.5
corner_only_homography = False

dot_grid_width = (dot_grid_cols - 1) * dot_spacing
dot_grid_height = (dot_grid_rows - 1) * dot_spacing


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


# Determine a rough lens distortion correction using the chessboard image
chess_config = calib.Config()
configured = chess_config.populate_lens_parameters_from_chessboard(
    chessboard_image_path, chess_grid_rows, chess_grid_cols
)

if not configured:
    print(
        "Could not determine lens correction properties from {}".format(
            chessboard_image_path
        )
    )
    exit()

params = cv.SimpleBlobDetector_Params()
params.minArea = 50
params.maxArea = 1000
params.filterByArea = True
params.minCircularity = 0.2
params.filterByCircularity = True
params.blobColor = 0
params.filterByColor = True
dot_detector = cv.SimpleBlobDetector_create(params)

# Use the chessboard derived lens correction to find the grid in the dots image
dot_grid_image = cv.imread(dot_grid_image_path)
undistorted_dot_image = calib.correct_image(
    dot_grid_image, chess_config, calib.Correction.lens_distortion
)

found, undistorted_grid = cv.findCirclesGrid(
    undistorted_dot_image,
    (dot_grid_cols, dot_grid_rows),
    cv.CALIB_CB_SYMMETRIC_GRID + cv.CALIB_CB_CLUSTERING,
    dot_detector,
    cv.CirclesGridFinderParameters(),
)

if not found:
    print("Could not find dot grid in {}".format(dot_grid_image_path))
    exit()

distorted_points = np.array(
    [[point] for point in cv.KeyPoint_convert(dot_detector.detect(dot_grid_image))],
    np.float32,
)
transformed_points = calib.correct_points(
    distorted_points, chess_config, calib.Correction.lens_distortion
)

# Use the lens-corrected grid and lens corrected points from the original image to determine the grid in the distorted image
# The lens distortion in the original image means that OpenCV cannot detect the grid itself
distorted_grid = np.zeros(undistorted_grid.shape, undistorted_grid.dtype)
for i in range(len(undistorted_grid)):
    if i % 100 == 0:
        print("progress: {}/{}".format(i, dot_grid_cols * dot_grid_rows), end="\r")
    # get the point at i in the grid
    grid_member = undistorted_grid[i]
    # find the nearest member of transformed_points, and its undistorted original
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
    assert original_point is not None
    # set that nearest, uncorrected point to position i in the new grid
    distorted_grid[i, 0, 0] = original_point[0, 0]
    distorted_grid[i, 0, 1] = original_point[0, 1]

# Determining lens correction properties with the full grid is too slow and memory intensive.
# Generate a sparser grid to improve speed and reduce RAM usage.
sparse_rows = dot_grid_rows // 2
sparse_cols = dot_grid_cols // 2
sparse_grid = np.zeros((sparse_rows * sparse_cols, 1, 2), np.float32)
for i in range(sparse_rows):
    for j in range(sparse_cols):
        sparse_grid[i * sparse_cols + j, 0, 0] = distorted_grid[
            (i * 2) * dot_grid_cols + (j * 2), 0, 0
        ]
        sparse_grid[i * sparse_cols + j, 0, 1] = distorted_grid[
            (i * 2) * dot_grid_cols + (j * 2), 0, 1
        ]

# Generate the calibration configuration using the newly determined grids
h, w = dot_grid_image.shape[:2]
dot_config = calib.Config()
configured = dot_config.populate_lens_parameters_from_grid(
    sparse_grid, sparse_cols, sparse_rows, w, h
)

if not configured:
    print(
        "Could not determine distortion properties from {}".format(dot_grid_image_path)
    )
    exit()

undistorted_distorted_grid = calib.correct_points(
    distorted_grid, dot_config, calib.Correction.lens_distortion
)
configured = dot_config.populate_keystone_and_real_parameters_from_grid(
    undistorted_distorted_grid,
    dot_grid_cols,
    dot_grid_rows,
    dot_grid_width,
    dot_grid_height,
    corners_only=corner_only_homography,
)

if not configured:
    print(
        "Could not determine homography properties from {}".format(dot_grid_image_path)
    )
    exit()

# Save the configuration to file
dot_config.save(calibration_file_name)
