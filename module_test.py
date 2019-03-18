import camera_calibration as calib
import cv2 as cv
import numpy as np

params = cv.SimpleBlobDetector_Params()
params.minArea = 50
params.maxArea = 1000
params.filterByArea = True
params.minCircularity = 0.2
params.filterByCircularity = True
params.blobColor = 0
params.filterByColor = True
detector = cv.SimpleBlobDetector_create(params)

finished, config = calib.Config.generate(
    'mocked checkboard.png', 10, 15,
    'cleaned_grids/distcor_01.bmp', detector, 116, 170, 85, 58
)

if not finished:
    print('unable to complete config')
    print(config)
    exit()

points = np.array([[[3584.902, 2468.0232]],
                   [[71.22837, 2466.539]],
                   [[68.2684, 62.64333]],
                   [[3600.0374, 78.32093]]], np.float32)

expectation = np.array([[85, 58],
                        [0, 58],
                        [0, 0],
                        [85, 0]], np.float32)

np.set_printoptions(suppress=True)
print('generated config:')
print(config)
print('image points:')
print(points)
print('calculated coordinates:')
print(calib.correct_points(points, config))
print('expected coordinates:')
print(expectation)

img = cv.imread('cleaned_grids/distcor_01.bmp')
img = calib.correct_camera_distortion(img, config)
img = calib.correct_keystone_distortion(img, config)

cv.imshow('grid', img)
cv.waitKey()
