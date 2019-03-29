import camera_calibration as calib
import cv2 as cv
import numpy as np


def test_separate_lens_and_keystone_image_correction_calls_are_equivalent_to_a_single_combined_call():

    image = cv.imread("sample_images/002h.bmp")

    config = calib.Config()
    config.populate_lens_parameters_from_chessboard(image, 6, 8)
    config.populate_keystone_and_real_parameters_from_chessboard(
        image, 8, 6, 90.06, 64.45
    )

    lens_corrected = calib.correct_image(
        image, config, calib.Correction.lens_distortion
    )
    keystone_corrected = calib.correct_image(
        lens_corrected, config, calib.Correction.keystone_distortion
    )

    combo_corrected = calib.correct_image(
        image, config, calib.Correction.lens_and_keystone
    )

    assert np.array_equal(keystone_corrected, combo_corrected)


def test_separate_lens_and_keystone_point_correction_calls_are_equivalent_to_a_single_combined_call():

    image = cv.imread("sample_images/002h.bmp")

    config = calib.Config()
    config.populate_lens_parameters_from_chessboard(image, 6, 8)
    config.populate_keystone_and_real_parameters_from_chessboard(
        image, 8, 6, 90.06, 64.45
    )

    _, points = cv.findChessboardCorners(image, (6, 8))

    lens_corrected = calib.correct_points(
        points, config, calib.Correction.lens_distortion
    )
    keystone_corrected = calib.correct_points(
        lens_corrected, config, calib.Correction.keystone_distortion
    )

    combo_corrected = calib.correct_points(
        points, config, calib.Correction.lens_and_keystone
    )

    assert np.array_equal(keystone_corrected, combo_corrected)


def test_separate_lens_keystone_and_real_point_correction_calls_are_equivalent_to_a_single_combined_call():

    image = cv.imread("sample_images/002h.bmp")

    config = calib.Config()
    config.populate_lens_parameters_from_chessboard(image, 6, 8)
    config.populate_keystone_and_real_parameters_from_chessboard(
        image, 8, 6, 90.06, 64.45
    )

    _, points = cv.findChessboardCorners(image, (6, 8))

    camera_corrected = calib.correct_points(
        points, config, calib.Correction.lens_and_keystone
    )
    real_converted = calib.correct_points(
        camera_corrected, config, calib.Correction.real_coordinates
    )

    combo_corrected = calib.correct_points(
        points, config, calib.Correction.lens_keystone_and_real_coordinates
    )

    assert np.array_equal(real_converted, combo_corrected)
