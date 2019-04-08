import camera_calibration as calib
import numpy as np
from tempfile import TemporaryFile
import sys


def test_empty_config_is_preserved_exactly_across_dictionary_conversion():
    config = calib.Config()

    dictionary = config.to_dict()
    loaded = calib.Config.from_dict(eval(repr(dictionary)))

    np.set_printoptions(threshold=sys.maxsize)

    assert loaded == config


def test_partial_config_is_preserved_exactly_across_dictionary_conversion():
    config = calib.Config()
    config.populate_lens_parameters_from_chessboard("sample_images/002h.bmp", 6, 8)

    dictionary = config.to_dict()
    loaded = calib.Config.from_dict(eval(repr(dictionary)))

    np.set_printoptions(threshold=sys.maxsize)

    assert loaded == config


def test_full_config_is_preserved_exactly_across_dictionary_conversion():
    config = calib.Config()
    config.populate_lens_parameters_from_chessboard("sample_images/002h.bmp", 6, 8)
    config.populate_keystone_and_real_parameters_from_chessboard(
        "sample_images/002h.bmp", 8, 6, 90.06, 64.45
    )

    dictionary = config.to_dict()
    loaded = calib.Config.from_dict(eval(repr(dictionary)))

    np.set_printoptions(threshold=sys.maxsize)

    assert loaded == config
