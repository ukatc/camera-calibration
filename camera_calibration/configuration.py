import attr
import cv2 as cv
import math
import numpy as np


@attr.s
class Corners:
    """
    Stores (X, Y) pairs for corners of a rectangle in an image
    """
    top_left = attr.ib()
    top_right = attr.ib()
    bottom_left = attr.ib()
    bottom_right = attr.ib()


@attr.s
class Config:
    """
    Camera calibration properties for a fixed camera looking at a given plane
    """
    distorted_camera_matrix = attr.ib(type=np.ndarray, default=None)
    distortion_coefficients = attr.ib(type=np.ndarray, default=None)
    undistorted_camera_matrix = attr.ib(type=np.ndarray, default=None)
    homography_matrix = attr.ib(type=np.ndarray, default=None)
    grid_image_corners = attr.ib(type=Corners, default=None)
    grid_space_corners = attr.ib(type=Corners, default=None)

    def populate_distortion_from_chessboard(self, chessboard_path: str, rows: int, cols: int):
        """
        Populate a config object's camera matrices and distortion coefficient properties using a chessboard image
        :param chessboard_path: The path to an image of a chess grid taken by the camera
        :param rows: The number of rows in the grid of black square corner intersection points
        :param cols: The number of columns in the grid of black square corner intersection points
        :return: A boolean indicating if the properties were successfully populated
        """
        chessboard = cv.imread(chessboard_path)
        gray_chessboard = cv.cvtColor(chessboard, cv.COLOR_BGR2GRAY)
        found, corners = cv.findChessboardCorners(gray_chessboard, (rows, cols))
        if not found:
            return False

        h, w = chessboard.shape[:2]
        return self.populate_distortion_from_grid(corners, rows, cols, w, h)

    def populate_distortion_from_symmetric_dot_pattern(self, dot_grid_path: str, dot_detector: cv.SimpleBlobDetector, rows: int, cols: int):
        """
        Populate a config object's camera matrices and distortion coefficient properties using a grid of reference dots
        :param dot_grid_path: The path to an image of a symmetric dot grid pattern taken by the camera
        :param dot_detector: An openCV dot detector able to detect all the dots in the given image
        :param rows: The number of rows in the dot grid
        :param cols: The number of columns in the dot grid
        :return: A boolean indicating if the properties were successfully populated
        """
        dots = cv.imread(dot_grid_path)
        found, grid = cv.findCirclesGrid(
            dots,
            (cols, rows),
            cv.CALIB_CB_SYMMETRIC_GRID + cv.CALIB_CB_CLUSTERING,
            dot_detector,
            cv.CirclesGridFinderParameters()
        )
        if not found:
            return False

        h, w = dots.shape[:2]
        return self.populate_distortion_from_grid(grid, rows, cols, w, h)

    def populate_homography_from_chessboard(self, chessboard_path: str, rows: int, cols: int, width: float, height: float):
        """
        Populate a config object's homography matrix and grid corner properties using a lens distorted chessboard image
        :param chessboard_path: The path to a chessboard image taken by the camera
        :param rows: The number of rows in the grid of black square corner intersection points
        :param cols: The number of columns in the grid of black square corner intersection points
        :param width: The width of the grid, measured from black square corner intersection points
        :param height: The height of the dot grid, measured from black square corner intersection points
        :return: A boolean indicating if the properties were successfully populated
        """
        chessboard = cv.imread(chessboard_path)
        gray_chessboard = cv.cvtColor(chessboard, cv.COLOR_BGR2GRAY)
        found, corners = cv.findChessboardCorners(gray_chessboard, (rows, cols))
        if not found:
            return False
        corners = cv.undistortPoints(corners,
                                     self.distorted_camera_matrix,
                                     self.distortion_coefficients,
                                     P=self.undistorted_camera_matrix)
        return self.populate_homography_from_grid(corners, rows, cols, width, height)

    def populate_homography_from_symmetric_dot_pattern(self, dot_grid_path: str, dot_detector: cv.SimpleBlobDetector, rows: int, cols: int, width: float, height: float):
        """
        Populate a config object's homography matrix and grid corner properties using a lens distorted grid of reference points
        :param dot_grid_path: The path to an image of a symmetric dot grid pattern taken by the camera
        :param dot_detector: An openCV dot detector able to detect all the dots in the given image
        :param rows: The number of rows in the dot grid
        :param cols: The number of columns in the dot grid
        :param width: The width of the dot grid, measured from dot centers
        :param height: The height of the dot grid, measured from dot centers
        :return: A boolean indicating if the properties were successfully populated
        """
        dots = cv.imread(dot_grid_path)
        delensed = cv.undistort(dots,
                                self.distorted_camera_matrix,
                                self.distortion_coefficients,
                                None,
                                self.undistorted_camera_matrix)
        found, grid = cv.findCirclesGrid(
            delensed,
            (cols, rows),
            cv.CALIB_CB_SYMMETRIC_GRID + cv.CALIB_CB_CLUSTERING,
            dot_detector,
            cv.CirclesGridFinderParameters()
        )

        if not found:
            return False

        return self.populate_homography_from_grid(grid, cols, rows, width, height)

    def populate_distortion_from_grid(self, grid: np.ndarray, rows: int, cols: int, image_width: int, image_height: int):
        """
        Populate a config object's camera matrices and distortion coefficient properties using a grid points
        :param grid: A grid of points from an image taken by the camera, generated by openCV's findCirclesGrid or findChessboardCorners methods
        :param rows: The number of rows in the grid
        :param cols: The number of columns in the grid
        :param image_width: The width of the camera's images in pixels
        :param image_height: The height of the camera's images in pixels
        :return: A boolean indicating if the properties were successfully populated
        """
        objp = np.zeros((cols * rows, 3), np.float32)
        objp[:, :2] = np.mgrid[0:rows * 5:5, 0:cols * 5:5].T.reshape(-1, 2)
        objpoints = [objp]
        imgpoints = [grid]

        calibrated, camera_matrix, distortion_coefficients, _, _ = cv.calibrateCamera(objpoints, imgpoints, (image_width, image_height), None,
                                                                                      None)

        if not calibrated:
            return False

        self.distorted_camera_matrix = camera_matrix
        self.distortion_coefficients = distortion_coefficients

        self.undistorted_camera_matrix, _ = cv.getOptimalNewCameraMatrix(camera_matrix,
                                                                         distortion_coefficients,
                                                                         (image_width, image_height),
                                                                         1,
                                                                         (image_width, image_height))
        return True

    def populate_homography_from_grid(self, grid: np.ndarray, cols: int, rows: int, width: float, height: float):
        """
        Populate a config object's homography matrix and grid corner properties using a grid of reference points which have been corrected for lens distortion
        :param grid: A grid of points from an image taken by the camera, generated by openCV's findCirclesGrid or findChessboardCorners methods, and corrected for lens distortion
        :param cols: The number of columns in the grid
        :param rows: The number of rows in the grid
        :param width: The distance from the leftmost to rightmost edges of the grid
        :param height: The distance from the topmost to bottommost edges of the grid
        :return: A boolean indicating if the properties were successfully populated
        """
        bottom_right = grid[0, 0]
        bottom_left = grid[cols - 1, 0]
        top_left = grid[-1, 0]
        top_right = grid[-cols, 0]

        def distance(p1, p2):
            return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

        border = 50
        grid_width = max(distance(top_left, top_right),
                         distance(bottom_left, bottom_right))
        grid_height = max(distance(top_left, bottom_left),
                          distance(top_right, bottom_right))
        point_spacing = math.ceil(max(grid_width / (cols - 1),
                                      grid_height / (rows - 1)))
        grid_width = point_spacing * (cols - 1)
        grid_height = point_spacing * (rows - 1)

        corners = np.array([top_left, top_right, bottom_left, bottom_right])
        target_corners = np.array([
            [border, border],  # top left
            [grid_width + border, border],  # top right
            [border, grid_height + border],  # bottom left
            [grid_width + border, grid_height + border],  # bottom right
        ])

        self.homography_matrix, _ = cv.findHomography(corners, target_corners)
        self.grid_image_corners = Corners(*target_corners)
        self.grid_space_corners = Corners((0, 0), (width, 0), (0, height),  (width, height))
        return True
