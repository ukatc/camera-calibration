import camera_calibration as calib
import cv2 as cv


def test_config_populated_from_paths_matches_config_populated_from_same_loaded_images():

    path = "sample_images/002h.bmp"

    path_config = calib.Config()
    assert path_config.populate_lens_parameters_from_chessboard(path, 6, 8)
    assert path_config.populate_keystone_and_real_parameters_from_chessboard(
        path, 8, 6, 90.06, 64.45
    )

    image = cv.imread(path)
    image_config = calib.Config()
    assert image_config.populate_lens_parameters_from_chessboard(image, 6, 8)
    assert image_config.populate_keystone_and_real_parameters_from_chessboard(
        image, 8, 6, 90.06, 64.45
    )

    assert path_config == image_config
