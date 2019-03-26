# Camera Calibration

A python library to simplify performing camera calibration and image un-distortion using OpenCV.

## Installation

The camera calibration package can be installed by running:
```bash
pip install git+https://github.com/ukatc/camera-calibration
```

## Usage

The package exposes a set of methods for correcting camera distortion in images and pixel
coordinates, as well as converting pixel coordinates to real world units.

These methods all require a calibration configuration object be passed in.
This configuration object can either have its attributes set directly, or via the various
`populate_*()` methods on the object, using a set of known reference points, or an image they can be
detected in.

```python
import camera_calibration as calib
import cv2

image_path = 'sample_images/002h.bmp'
rows = 6
cols = 8

config = calib.Config()
config.populate_distortion_from_chessboard(image_path, rows, cols)
config.populate_homography_from_chessboard(image_path, cols, rows, 90.06, 64.45)

bgr = cv2.imread(image_path)
undistorted = calib.correct_camera_distortion(bgr, config)
cv2.imwrite('undistorted.png', undistorted)
grid_aligned = calib.correct_keystone_distortion(undistorted, config)
cv2.imwrite('grid_aligned.png', grid_aligned)
```

## Dev environment

The dev environment can be set up by running
```bash
pip install -r requirements.txt
```

### Tests

Tests are performed using pytest, which is included in the packages's `requirements.txt`.
They can be run from the package's directory with the `pytest` command

## Versioning

This package uses [semantic versioning](https://semver.org/).
