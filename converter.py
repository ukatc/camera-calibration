"""
Reads a chessboard image from a camera to obtain calibration data.
That data is then used to transform pixel coordinates in other images from that camera to real world positions in millimeters
"""

import numpy as np
import cv2
import math

side_length = 5  # The sides of the chessboard squares in millimeters
row_corners = 15  # The number of corners on a row in the chessboard image
col_corners = 10  # The number of corners on a column in the chessboard image
img = cv2.imread('mocked checkboard.png')  # The chessboard calibration image

objp = np.zeros((row_corners * col_corners, 3), np.float32)
objp[:, :2] = np.mgrid[0:col_corners*side_length:side_length, 0:row_corners*side_length:side_length].T.reshape(-1, 2)

grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
found, corners = cv2.findChessboardCorners(grayscale, (col_corners, row_corners))

if not found:
    print('Unable to find chessboard in image')
    exit()

h, w = img.shape[:2]
calibrated, camera_matrix, distortion_coefficients, rotate_vector, translate_vector = cv2.calibrateCamera([objp], [corners], (w, h), None, None)

translate_vector = translate_vector[0]
rotate_vector = rotate_vector[0]

if not calibrated:
    print('Unable to calibrate camera')
    exit()


print('Camera matrix:\n' + str(camera_matrix))
print('Rotation vector:\n' + str(rotate_vector))
print('Translation vector:\n' + str(translate_vector))

rotate_matrix, _ = cv2.Rodrigues(rotate_vector)


def groundprojectpoint(image_point, z=0.0):
    '''
    Projects a point in the image onto the given z-plane
    Resulting coordinates are given in millimeters
    :param image_point: a point on the camera pane as a (u,v) pair
    :param z: the z plane to project the image point onto
    :return: the points coordinates in 3D space
    '''
    inverse_rotate = np.linalg.inv(rotate_matrix)
    inverse_camera = np.linalg.inv(camera_matrix)

    uv_point = np.ones((3, 1))
    uv_point[0][0] = image_point[0]
    uv_point[1][0] = image_point[1]

    temp_matrix = np.matmul(np.matmul(inverse_rotate, inverse_camera), uv_point)
    temp_matrix2 = np.matmul(inverse_rotate, translate_vector)

    s = (z + temp_matrix2[2, 0]) / temp_matrix[2, 0]
    wc_point = np.matmul(inverse_rotate, (np.matmul(s * inverse_camera, uv_point) - translate_vector))

    wc_point[2] = z

    return wc_point


def testpoints(point1, point2, expecteddistance):
    a = groundprojectpoint(point1)
    b = groundprojectpoint(point2)
    distance = math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    error = (abs(expecteddistance - distance) / expecteddistance) * 100
    print('expected: {} got: {} error:{:.2f}%'.format(expecteddistance, distance, error))


testpoints((68, 62), (72, 2466), 115/2)  # distcor_01 - topleft to bottomleft
testpoints((68, 62), (88, 62), 1/2)  # distcor_01 - topleft to right neighbour
testpoints((68, 62), (3600, 78), 169/2)  # distcor_01 - topleft to topright
testpoints((1800, 1283), (1822, 1283), 1/2)  # distcor_01 - horizontal neighbours near center
