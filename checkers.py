"""
Camera calibration script for checkerboard patterns

Based on https://docs.opencv.org/3.4.3/dc/dbb/tutorial_py_calibration.html
"""
import numpy as np
import cv2 as cv

row_corners = 15
col_corners = 10

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....
objp = np.zeros((row_corners * col_corners, 3), np.float32)
objp[:,:2] = np.mgrid[0:col_corners, 0:row_corners].T.reshape(-1, 2)
# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

img = cv.imread('mocked checkboard.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# Find the chess board corners
ret, corners = cv.findChessboardCorners(gray, (col_corners, row_corners), None)
# If found, add object points, image points (after refining them)
if ret:
    objpoints.append(objp)
    imgpoints.append(corners)

    h, w = img.shape[:2]
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (w, h), None, None)

    img = cv.imread('distcor_01-edited.bmp')
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # undistort
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)

    print(mtx)
    print(newcameramtx)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv.imwrite('calibresult.png', dst)

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error
    print("total error: {}".format(mean_error / len(objpoints)))

    cv.waitKey()
