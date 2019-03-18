import attr
import cv2 as cv
import math
import numpy as np


@attr.s
class Corners:
    top_left = attr.ib()
    top_right = attr.ib()
    bottom_left = attr.ib()
    bottom_right = attr.ib()


@attr.s
class Config:
    distorted_camera_matrix = attr.ib(type=np.ndarray, default=None)
    distortion_coefficients = attr.ib(type=np.ndarray, default=None)
    undistorted_camera_matrix = attr.ib(type=np.ndarray, default=None)
    homography_matrix = attr.ib(type=np.ndarray, default=None)
    grid_image_corners = attr.ib(type=Corners, default=None)
    grid_space_corners = attr.ib(type=Corners, default=None)

    @staticmethod
    def generate(
            chessboard_path, chess_rows, chess_cols,
            dot_grid_path, dot_detector, dot_rows, dot_cols, dot_grid_width, dot_grid_height
    ):

        config = Config()

        chessboard = cv.imread(chessboard_path)
        gray_chessboard = cv.cvtColor(chessboard, cv.COLOR_BGR2GRAY)
        found, corners = cv.findChessboardCorners(gray_chessboard, (chess_rows, chess_cols))
        if not found:
            return False, config

        objp = np.zeros((chess_cols * chess_rows, 3), np.float32)
        objp[:, :2] = np.mgrid[0:chess_rows*5:5, 0:chess_cols*5:5].T.reshape(-1, 2)
        objpoints = [objp]
        imgpoints = [corners]

        h, w = chessboard.shape[:2]
        calibrated, camera_matrix, distortion_coefficients, _, _ = cv.calibrateCamera(objpoints, imgpoints, (w, h), None, None)

        if not calibrated:
            return False, config

        config.distorted_camera_matrix = camera_matrix
        config.distortion_coefficients = distortion_coefficients

        new_camera_matrix, _ = cv.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients, (w, h), 1, (w, h))

        config.undistorted_camera_matrix = new_camera_matrix

        dots = cv.imread(dot_grid_path)
        delensed = cv.undistort(dots, camera_matrix, distortion_coefficients, None, new_camera_matrix)
        found, grid = cv.findCirclesGrid(
            delensed,
            (dot_cols, dot_rows),
            cv.CALIB_CB_SYMMETRIC_GRID + cv.CALIB_CB_CLUSTERING,
            dot_detector,
            cv.CirclesGridFinderParameters()
        )

        if not found:
            return False, config

        bottom_right = grid[0, 0]
        bottom_left = grid[dot_cols - 1, 0]
        top_left = grid[-1, 0]
        top_right = grid[-dot_cols, 0]

        def distance(p1, p2):
            return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

        border = 50
        grid_width = max(distance(top_left, top_right),
                         distance(bottom_left, bottom_right))
        grid_height = max(distance(top_left, bottom_left),
                          distance(top_right, bottom_right))
        point_spacing = math.ceil(max(grid_width / (dot_cols - 1),
                                      grid_height / (dot_rows - 1)))
        grid_width = point_spacing * (dot_cols - 1)
        grid_height = point_spacing * (dot_rows - 1)

        corners = np.array([top_left, top_right, bottom_left, bottom_right])
        target_corners = np.array([
            [border, border],  # top left
            [grid_width + border, border],  # top right
            [border, grid_height + border],  # bottom left
            [grid_width + border, grid_height + border],  # bottom right
        ])

        config.homography_matrix, _ = cv.findHomography(corners, target_corners)
        config.grid_image_corners = Corners(*target_corners)
        config.grid_space_corners = Corners((0, 0), (dot_grid_width, 0), (0, dot_grid_height), (dot_grid_width, dot_grid_height))

        return True, config
