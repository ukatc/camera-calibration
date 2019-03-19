import camera_calibration as calib
import numpy as np
import pytest


def assess_points_transform_to_given_absolute_accuracy(accuracy: float):
    """
    Apply a camera correction to points, testing if the result is within the given accuracy of their expected corrected
    values
    :param accuracy: A point will be accepted if its x and y components are this close to their expected values
    """
    np.set_printoptions(suppress=True)
    sample_config = calib.Config(distorted_camera_matrix=np.array([[3.2894140643482078e+03, 0.0000000000000000e+00, 1.9176779735640416e+03],
                                                                   [0.0000000000000000e+00, 3.2904025933090470e+03, 1.3998964108727967e+03],
                                                                   [0.0000000000000000e+00, 0.0000000000000000e+00, 1.0000000000000000e+00]]),
                                 distortion_coefficients=np.array([[-0.091678368932376 ,  0.004685276554469 , -0.002864536969103 , -0.0007950003034298,  0.0239023771408992]]),
                                 undistorted_camera_matrix=np.array([[3.1303671875000000e+03, 0.0000000000000000e+00, 1.9138636096939445e+03],
                                                                     [0.0000000000000000e+00, 3.1353242187500000e+03, 1.3911821821740596e+03],
                                                                     [0.0000000000000000e+00, 0.0000000000000000e+00, 1.0000000000000000e+00]]),
                                 homography_matrix=np.array([[ 1.0052048861138621e+00, -2.2537481739086729e-03, -3.0176134059966238e+01],
                                                             [-5.6034672817510825e-03,  1.0057402244836438e+00, -1.8547576373566425e+01],
                                                             [-2.5280366141280606e-06, -2.6176789860478385e-06, 1.0000000000000000e+00]]),
                                 grid_image_corners=calib.Corners(top_left=np.array([50, 50]), top_right=np.array([3599,   50]), bottom_left=np.array([  50, 2465]), bottom_right=np.array([3599, 2465])),
                                 grid_space_corners=calib.Corners(top_left=(0, 0), top_right=(85, 0), bottom_left=(0, 58), bottom_right=(85, 58)))

    # Points determined by dot centers from running an openCV blob detector over cleaned_grids/distcor_01.bmp
    points = np.array([[[3584.902, 2468.0232]],  # bottom right
                       [[71.22837, 2466.539]],  # bottom left
                       [[68.2684, 62.64333]],  # top left
                       [[3600.0374, 78.32093]],  # top right

                       [[1804.8428, 38.65753]],  # middle top
                       [[1799.092, 2498.543]],  # middle bottom
                       [[47.950756, 1299.2955]],  # middle left
                       [[3611.6602, 1307.9681]],  # middle right
                       [[1800.4049, 1304.3975]],  # center
                       ], np.float32)

    expectations = np.array([[85, 58],
                             [0, 58],
                             [0, 0],
                             [85, 0],

                             [83/2, 0],
                             [83/2, 58],
                             [0, 59/2],
                             [85, 59/2],
                             [83/2, 59/2],
                             ], np.float32)

    corrected_points = calib.correct_points(points, sample_config)

    assert points.shape == corrected_points.shape

    print(expectations)
    print(np.array([point[0] for point in corrected_points]))

    for i in range(len(corrected_points)):
        original = points[i, 0]
        point = corrected_points[i, 0]
        expectation = expectations[i]
        assert len(point) == 2

        assert point == pytest.approx(expectation, abs=accuracy), \
            'point {} corrects to {}. Expected values within {}mm of {}'.format(original, point, accuracy, expectation)


def test_points_transform_accurate_to_nearest_mm():
    assess_points_transform_to_given_absolute_accuracy(0.5)


def test_points_transform_accurate_to_nearest_tenth_mm():
    assess_points_transform_to_given_absolute_accuracy(0.05)


def test_points_transform_accurate_to_nearest_hundredth_mm():
    assess_points_transform_to_given_absolute_accuracy(0.005)


@pytest.mark.xfail  # Test for micron accuracy, but don't expect it
def test_points_transform_accurate_to_nearest_micron():
    assess_points_transform_to_given_absolute_accuracy(0.0005)
