import time
import cv2
import numpy as np


bgr = cv2.imread('cleaned_grids/distcor_01.bmp')

greyscale = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(greyscale, (5, 5), 0)
retval, thresholded = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imwrite('binarized.png', thresholded)

params = cv2.SimpleBlobDetector_Params()
params.minArea = 50
params.maxArea = 1000
params.filterByArea = True
params.minCircularity = 0.2
params.filterByCircularity = True
params.blobColor = 0
params.filterByColor = True

print('detecting points')
detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(bgr)
points = cv2.KeyPoint_convert(keypoints)
print('{} points found'.format(len(points)))

print('searching for grid - ' + time.strftime('%H:%M:%S', time.gmtime()))
gridparams = cv2.CirclesGridFinderParameters()

# Based on https://docs.opencv.org/3.4.3/dc/dbb/tutorial_py_calibration.html

grid_cols, grid_rows = (170, 116)  # 19720 expected points

found, grid = cv2.findCirclesGrid(bgr, (grid_cols, grid_rows), cv2.CALIB_CB_SYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING, detector, gridparams)
griddoodle = bgr.copy()
cv2.drawChessboardCorners(griddoodle, (grid_cols, grid_rows), grid if found else points, found)
cv2.imwrite('detected_points.bmp', griddoodle)
cv2.imshow('original', bgr)
cv2.imshow('grid points', griddoodle)
if found:
    print(grid)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(44,27,0)
    objp = np.zeros((grid_rows * grid_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:grid_cols, 0:grid_rows].T.reshape(-1, 2)
    print(objp)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    objpoints.append(objp)
    imgpoints.append(grid)

    h, w = bgr.shape[:2]
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    undistorted = cv2.undistort(bgr, mtx, dist, None, newcameramtx)
    cv2.imshow('undistorted', undistorted)

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    print("total error: {}".format(mean_error / len(objpoints)))
    mean_error = 0

else:
    print('grid not found')
print(time.strftime('%H:%M:%S', time.gmtime()))
cv2.waitKey()
