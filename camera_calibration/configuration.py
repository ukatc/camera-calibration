import attr
import cv2 as cv
import math
import numpy as np
from typing import Optional, Union


@attr.s(cmp=False)
class Corners(object):
    """
    Stores (X, Y) pairs for corners of a rectangle in an image as 2 item numpy arrays
    """

    top_left = attr.ib(type=np.ndarray)
    top_right = attr.ib(type=np.ndarray)
    bottom_left = attr.ib(type=np.ndarray)
    bottom_right = attr.ib(type=np.ndarray)

    def __eq__(self, other):
        """
        Compare config objects for equality.

        Needed as numpy arrays are don't implement the python __eq__ spec.
        See https://github.com/python-attrs/attrs/issues/435 for discussion
        """
        return (
            isinstance(other, Corners)
            and np.array_equal(self.top_left, other.top_left)
            and np.array_equal(self.top_right, other.top_right)
            and np.array_equal(self.bottom_left, other.bottom_left)
            and np.array_equal(self.bottom_right, other.bottom_right)
        )


@attr.s(cmp=False)
class Config(object):
    """
    Camera calibration properties for a fixed camera looking at a given plane

    Distortion related attributes:
        distorted_camera_matrix: A 3x3 numpy array representing the camera matrix
        distortion_coefficients: A 1x(4, 5, 8, 12, or 14) numpy array containing the camera's distortion coefficients
        undistorted_camera_matrix: A 3x3 numpy array containing the new camera matrix for projecting undistorted images
    Keystone/Homography related attributes:
        homography_matrix: A 3x3 numpy array containing a homography matrix for projecting lens distortion corrected images to ones where the calibration grid is made of identical rectangles
        grid_image_corners: A Corners instance containing the calibration grids corners as coordinates in a lens and keystone distortion corrected image
        grid_space_corners: A Corners instance containing the calibration grids corners as coordinates on the plane in the real world
    """

    distorted_camera_matrix = attr.ib(type=np.ndarray, default=None)
    distortion_coefficients = attr.ib(type=np.ndarray, default=None)
    undistorted_camera_matrix = attr.ib(type=np.ndarray, default=None)
    homography_matrix = attr.ib(type=np.ndarray, default=None)
    grid_image_corners = attr.ib(type=Corners, default=None)
    grid_space_corners = attr.ib(type=Corners, default=None)

    def __eq__(self, other):
        """
        Compare config objects for equality.

        Needed as numpy arrays are don't implement the python __eq__ spec.
        See https://github.com/python-attrs/attrs/issues/435 for discussion
        """
        return (
            isinstance(other, Config)
            and np.array_equal(
                self.distorted_camera_matrix, other.distorted_camera_matrix
            )
            and np.array_equal(
                self.distortion_coefficients, other.distortion_coefficients
            )
            and np.array_equal(
                self.undistorted_camera_matrix, other.undistorted_camera_matrix
            )
            and np.array_equal(self.homography_matrix, other.homography_matrix)
            and (
                (self.grid_image_corners is None and other.grid_image_corners is None)
                or (
                    self.grid_image_corners is not None
                    and other.grid_image_corners is not None
                    and self.grid_image_corners == other.grid_image_corners
                )
            )
            and (
                (self.grid_space_corners is None and other.grid_space_corners is None)
                or (
                    self.grid_space_corners is not None
                    and other.grid_space_corners is not None
                    and self.grid_space_corners == other.grid_space_corners
                )
            )
        )

    def populate_lens_parameters_from_chessboard(self, chessboard_image, rows, cols):
        # type: (Union[np.ndarray, str], int, int) -> bool
        """
        Populate a config object's camera matrices and distortion coefficient properties using a chessboard image
        :param chessboard_image: An image, or path to an image, of a chess grid taken by the camera
        :param rows: The number of rows in the grid of black square corner intersection points
        :param cols: The number of columns in the grid of black square corner intersection points
        :return: A boolean indicating if the properties were successfully populated
        """
        chessboard = get_image(chessboard_image)
        gray_chessboard = cv.cvtColor(chessboard, cv.COLOR_BGR2GRAY)
        found, corners = cv.findChessboardCorners(gray_chessboard, (rows, cols))
        if not found:
            return False

        h, w = chessboard.shape[:2]
        return self.populate_lens_parameters_from_grid(corners, rows, cols, w, h)

    def populate_lens_parameters_from_symmetric_dot_pattern(
        self, dot_grid_image, dot_detector, rows, cols
    ):
        # type: (Union[np.ndarray, str], cv.SimpleBlobDetector, int, int) -> bool
        """
        Populate a config object's camera matrices and distortion coefficient properties using a grid of reference dots
        :param dot_grid_image: An image, or path to an image, of a symmetric dot grid pattern taken by the camera
        :param dot_detector: An openCV dot detector able to detect all the dots in the given image
        :param rows: The number of rows in the dot grid
        :param cols: The number of columns in the dot grid
        :return: A boolean indicating if the properties were successfully populated
        """
        dots = get_image(dot_grid_image)
        found, grid = cv.findCirclesGrid(
            dots,
            (cols, rows),
            cv.CALIB_CB_SYMMETRIC_GRID + cv.CALIB_CB_CLUSTERING,
            dot_detector,
            cv.CirclesGridFinderParameters(),
        )
        if not found:
            return False

        h, w = dots.shape[:2]
        return self.populate_lens_parameters_from_grid(grid, rows, cols, w, h)

    def populate_keystone_and_real_parameters_from_chessboard(
        self, chessboard_image, rows, cols, width, height, corners_only=False, border=50
    ):
        # type: (Union[np.ndarray, str], int, int, float, float, bool, int) -> bool
        """
        Populate a config object's homography matrix and grid corner properties using a lens distorted chessboard image
        :param chessboard_image: An image, or path to an image, of a chessboard image taken by the camera
        :param rows: The number of rows in the grid of black square corner intersection points
        :param cols: The number of columns in the grid of black square corner intersection points
        :param width: The width of the grid, measured from black square corner intersection points
        :param height: The height of the dot grid, measured from black square corner intersection points
        :param corners_only: Whether to calculate the transform with just the corner points, or all points of the grid
        :param border: The size of the border in pixels to place around the chessboard's area in a keystone corrected image
        :return: A boolean indicating if the properties were successfully populated
        """
        chessboard = get_image(chessboard_image)
        gray_chessboard = cv.cvtColor(chessboard, cv.COLOR_BGR2GRAY)
        found, corners = cv.findChessboardCorners(gray_chessboard, (rows, cols))
        if not found:
            return False
        corners = cv.undistortPoints(
            corners,
            self.distorted_camera_matrix,
            self.distortion_coefficients,
            P=self.undistorted_camera_matrix,
        )
        return self.populate_keystone_and_real_parameters_from_grid(
            corners, rows, cols, width, height, corners_only, border
        )

    def populate_keystone_and_real_parameters_from_symmetric_dot_pattern(
        self,
        dot_grid_image,
        dot_detector,
        rows,
        cols,
        width,
        height,
        corners_only=False,
        border=50,
    ):
        # type: (Union[np.ndarray, str], cv.SimpleBlobDetector, int, int, float, float, bool, int) -> bool
        """
        Populate a config object's homography matrix and grid corner properties using a lens distorted grid of reference points
        :param dot_grid_image: An image, or path to an image, of a symmetric dot grid pattern taken by the camera
        :param dot_detector: An openCV dot detector able to detect all the dots in the given image
        :param rows: The number of rows in the dot grid
        :param cols: The number of columns in the dot grid
        :param width: The width of the dot grid, measured from dot centers
        :param height: The height of the dot grid, measured from dot centers
        :param corners_only: Whether to calculate the transform with just the corner points, or all points of the grid
        :param border: The size of the border in pixels to place around the dot grid's area in a keystone corrected image
        :return: A boolean indicating if the properties were successfully populated
        """
        dots = get_image(dot_grid_image)
        delensed = cv.undistort(
            dots,
            self.distorted_camera_matrix,
            self.distortion_coefficients,
            None,
            self.undistorted_camera_matrix,
        )
        found, grid = cv.findCirclesGrid(
            delensed,
            (cols, rows),
            cv.CALIB_CB_SYMMETRIC_GRID + cv.CALIB_CB_CLUSTERING,
            dot_detector,
            cv.CirclesGridFinderParameters(),
        )

        if not found:
            return False

        return self.populate_keystone_and_real_parameters_from_grid(
            grid, cols, rows, width, height, corners_only, border
        )

    def populate_lens_parameters_from_grid(
        self, grid, rows, cols, image_width, image_height
    ):
        # type: (np.ndarray, int, int, int, int) -> bool
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
        objp[:, :2] = np.mgrid[0 : rows * 5 : 5, 0 : cols * 5 : 5].T.reshape(-1, 2)
        objpoints = [objp]
        imgpoints = [grid]

        calibrated, camera_matrix, distortion_coefficients, _, _ = cv.calibrateCamera(
            objpoints, imgpoints, (image_width, image_height), None, None
        )

        if not calibrated:
            return False

        self.distorted_camera_matrix = camera_matrix
        self.distortion_coefficients = distortion_coefficients

        self.undistorted_camera_matrix, _ = cv.getOptimalNewCameraMatrix(
            camera_matrix,
            distortion_coefficients,
            (image_width, image_height),
            1,
            (image_width, image_height),
        )
        return True

    def populate_keystone_and_real_parameters_from_grid(
        self, grid, cols, rows, width, height, corners_only=False, border=50
    ):
        # type: (np.ndarray, int, int, float, float, bool, int) -> bool
        """
        Populate a config object's homography matrix and grid corner properties using a grid of reference points which have been corrected for lens distortion
        :param grid: A grid of points from an image taken by the camera, generated by openCV's findCirclesGrid or findChessboardCorners methods, and corrected for lens distortion
        :param cols: The number of columns in the grid
        :param rows: The number of rows in the grid
        :param width: The distance from the leftmost to rightmost edges of the grid
        :param height: The distance from the topmost to bottommost edges of the grid
        :param corners_only: Whether to calculate the transform with just the corner points, or all points of the grid
        :param border: The size of the border in pixels to place around the grid's area in a keystone corrected image
        :return: A boolean indicating if the properties were successfully populated
        """
        bottom_right = grid[0, 0]
        bottom_left = grid[cols - 1, 0]
        top_left = grid[-1, 0]
        top_right = grid[-cols, 0]

        def distance(p1, p2):
            return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

        grid_width = max(
            distance(top_left, top_right), distance(bottom_left, bottom_right)
        )
        grid_height = max(
            distance(top_left, bottom_left), distance(top_right, bottom_right)
        )
        point_spacing = math.ceil(
            max(grid_width / (cols - 1), grid_height / (rows - 1))
        )
        grid_width = point_spacing * (cols - 1)
        grid_height = point_spacing * (rows - 1)

        target_corners = np.array(
            [
                [border, border],  # top left
                [grid_width + border, border],  # top right
                [border, grid_height + border],  # bottom left
                [grid_width + border, grid_height + border],  # bottom right
            ]
        )

        if corners_only:
            grid_points = np.array([top_left, top_right, bottom_left, bottom_right])
            target_points = target_corners
        else:
            grid_points = np.array([dot[0] for dot in grid])
            target_points = np.zeros((rows * cols, 2), np.float32)
            # measuring from the bottom left, initially going along rows, to match the detected dot grid
            for row in range(rows):
                for col in range(cols):
                    target_points[row * cols + col, :2] = (
                        (border + ((cols - (1 + col)) * point_spacing)),
                        (border + ((rows - (1 + row)) * point_spacing)),
                    )

        self.homography_matrix, _ = cv.findHomography(grid_points, target_points)
        self.grid_image_corners = Corners(*target_corners)
        self.grid_space_corners = Corners(
            np.array((0, 0), np.float32),
            np.array((width, 0), np.float32),
            np.array((0, height), np.float32),
            np.array((width, height), np.float32),
        )
        return True

    @staticmethod
    def load(file):
        """
        Loads a saved camera calibration configuration from file
        :param file: The file, or name of the file, to load the configuration from
        :return: A Config object populated with the loaded values
        """
        npz_file = np.load(file)

        config = Config()
        if "distorted_camera_matrix" in npz_file:
            array = npz_file["distorted_camera_matrix"]
            if array.shape == (3, 3):
                config.distorted_camera_matrix = array
        if "distortion_coefficients" in npz_file:
            array = npz_file["distortion_coefficients"]
            if array.shape[0] == 1 and array.shape[1] in (4, 5, 8, 12, 14):
                config.distortion_coefficients = array
        if "undistorted_camera_matrix" in npz_file:
            array = npz_file["undistorted_camera_matrix"]
            if array.shape == (3, 3):
                config.undistorted_camera_matrix = array
        if "homography_matrix" in npz_file:
            array = npz_file["homography_matrix"]
            if array.shape == (3, 3):
                config.homography_matrix = array
        if "grid_image_corners" in npz_file:
            array = npz_file["grid_image_corners"]
            if array.shape == (4, 2):
                config.grid_image_corners = corners_from_array(array)
        if "grid_space_corners" in npz_file:
            array = npz_file["grid_space_corners"]
            if array.shape == (4, 2):
                config.grid_space_corners = corners_from_array(array)

        npz_file.close()
        return config

    def save(self, file):
        """
        Save the current state of the config object to file
        :param file: The file, or name of the file, to save the config's values to
        """
        np.savez(
            file,
            distorted_camera_matrix=self.distorted_camera_matrix,
            distortion_coefficients=self.distortion_coefficients,
            undistorted_camera_matrix=self.undistorted_camera_matrix,
            homography_matrix=self.homography_matrix,
            grid_image_corners=corners_to_array(self.grid_image_corners),
            grid_space_corners=corners_to_array(self.grid_space_corners),
        )

    @staticmethod
    def from_dict(dictionary):
        """
        Generate a camera calibration configuration from a dictionary generated with the to_dict() method
        :param dictionary: The dictionary to load the configuration values from
        :return: A Config object populated with the loaded values
        """
        return Config(
            array_for_list(dictionary["distorted_camera_matrix"]),
            array_for_list(dictionary["distortion_coefficients"]),
            array_for_list(dictionary["undistorted_camera_matrix"]),
            array_for_list(dictionary["homography_matrix"]),
            corners_from_dict(dictionary["grid_image_corners"]),
            corners_from_dict(dictionary["grid_space_corners"]),
        )

    def to_dict(self):
        """
        Generate a dictionary containing list copies of the config's attributes
        :return: The dictionary instance
        """
        return {
            "distorted_camera_matrix": list_for_array(self.distorted_camera_matrix),
            "distortion_coefficients": list_for_array(self.distortion_coefficients),
            "undistorted_camera_matrix": list_for_array(self.undistorted_camera_matrix),
            "homography_matrix": list_for_array(self.homography_matrix),
            "grid_image_corners": corners_to_dict(self.grid_image_corners),
            "grid_space_corners": corners_to_dict(self.grid_space_corners),
        }


def corners_to_array(corners):
    # type: (Optional[Corners]) -> Optional[np.ndarray]
    if corners is None:
        return None
    return np.array(
        [
            corners.top_left,
            corners.top_right,
            corners.bottom_left,
            corners.bottom_right,
        ],
        np.float32,
    )


def corners_from_array(array):
    # type: (np.ndarray) -> Corners
    if array.shape != (4, 2):
        raise ValueError(
            "Array shape was not the required (4, 2) to populate a Corners object"
        )
    return Corners(
        top_left=array[0].copy(),
        top_right=array[1].copy(),
        bottom_left=array[2].copy(),
        bottom_right=array[3].copy(),
    )


def corners_to_dict(corners):
    # type: (Optional[Corners]) -> Optional[dict]
    if corners is None:
        return None
    return {
        "top_left": list_for_array(corners.top_left),
        "top_right": list_for_array(corners.top_right),
        "bottom_left": list_for_array(corners.bottom_left),
        "bottom_right": list_for_array(corners.bottom_right),
    }


def corners_from_dict(dictionary):
    # type: (Optional[dict]) -> Optional[Corners]
    if dictionary is None:
        return None
    return Corners(
        array_for_list(dictionary["top_left"]),
        array_for_list(dictionary["top_right"]),
        array_for_list(dictionary["bottom_left"]),
        array_for_list(dictionary["bottom_right"]),
    )


def list_for_array(array):
    # type: (Optional[np.ndarray]) -> Optional[list]
    if array is None:
        return None
    return list(array.tolist())


def array_for_list(input_list):
    # type: (Optional[list]) -> Optional[np.ndarray]
    if input_list is None:
        return None
    return np.array(input_list)


def get_image(img):
    # type: (Union[np.ndarray, str]) -> np.ndarray
    """
    Return an image either by returning the argument if it's already a numpy array, or loading it as one using openCV
    :param img: An image, or path to an image
    :return: The image
    """
    if isinstance(img, np.ndarray):
        return img
    else:
        return cv.imread(img)
