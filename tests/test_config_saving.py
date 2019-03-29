import camera_calibration as calib
import numpy as np
from tempfile import TemporaryFile
import sys


def test_partial_config_is_preserved_exactly_across_save_and_load():
    file = TemporaryFile()

    config = calib.Config()
    config.populate_lens_parameters_from_chessboard("sample_images/002h.bmp", 6, 8)

    config.save(file)
    file.seek(0)
    loaded = calib.Config.load(file)

    np.set_printoptions(threshold=sys.maxsize)

    assert loaded == config


def test_full_config_is_preserved_exactly_across_save_and_load():
    file = TemporaryFile()

    config = calib.Config()
    config.populate_lens_parameters_from_chessboard("sample_images/002h.bmp", 6, 8)
    config.populate_keystone_and_real_parameters_from_chessboard(
        "sample_images/002h.bmp", 8, 6, 90.06, 64.45
    )

    config.save(file)
    file.seek(0)
    loaded = calib.Config.load(file)

    np.set_printoptions(threshold=sys.maxsize)

    assert loaded == config
