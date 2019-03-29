"""
Analyse the saved calibration produced by 'store_precise_calibration.py'
This is done by detecting points in the calibration image used and calculating their real world position.
The differences between the calculated real world point, and the nearest grid point are then used to analyse the
config.

Analysis is made of:
- The maximum deviation
- The average deviation
- A histogram of deviations, rounded up to the nearest micron
- A heatmap of deviations by grid point, with brighter red marking larger deviation

This is all based on the assumption that the corrections are accurate to the nearest point in a 0.5mm grid
"""
import camera_calibration as calib
import cv2 as cv
import math
import numpy as np
from pprint import pprint

calibration_file_name = "calibration.npz"
dot_grid_image_path = "../sample_images/distcor_04_cleaned.bmp"
heatmap_image_file_name = "heatmap.bmp"
dot_grid_rows = 116
dot_grid_cols = 170


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


dot_config = calib.Config.load(calibration_file_name)

params = cv.SimpleBlobDetector_Params()
params.minArea = 50
params.maxArea = 1000
params.filterByArea = True
params.minCircularity = 0.2
params.filterByCircularity = True
params.blobColor = 0
params.filterByColor = True
dot_detector = cv.SimpleBlobDetector_create(params)

dot_image = cv.imread(dot_grid_image_path)
distorted_points = np.array(
    [[point] for point in cv.KeyPoint_convert(dot_detector.detect(dot_image))],
    np.float32,
)

corrected_points = calib.correct_points(
    distorted_points, dot_config, calib.Correction.lens_keystone_and_real_coordinates
)

distances = []
distance_hist = {}
deviation_map = np.zeros((dot_grid_cols, dot_grid_rows), np.float32)

# Determine each corrected points distance from the nearest point in a 0.5mm square grid
for lone_point in corrected_points:
    point = lone_point[0]
    x = int(round(point[0] * 2))
    y = int(round(point[1] * 2))
    expectation = (x / 2.0, y / 2.0)
    deviation = distance(point, expectation)
    distances.append(deviation)
    deviation_map[x, y] = deviation
    microns = math.ceil(deviation * 1000)
    if microns in distance_hist:
        distance_hist[microns] += 1
    else:
        distance_hist[microns] = 1

max_deviation = max(distances)

print("average deviation:\n{}mm".format(sum(distances) / len(distances)))
print("max deviation:\n{}mm".format(max_deviation))
print("deviation spread:")
pprint(distance_hist)

heatmap = np.zeros((dot_grid_rows, dot_grid_cols, 3), np.uint8)
for i in range(dot_grid_cols):
    for j in range(dot_grid_rows):
        heatmap[j, i, 2] = 255 * (deviation_map[i, j] / max_deviation)
cv.imwrite(heatmap_image_file_name, heatmap)
