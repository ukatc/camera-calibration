# Camera Calibration


![version](https://img.shields.io/github/release-pre/ukatc/camera-calibration.svg)
![python versions](https://img.shields.io/badge/python-2.7%20|%203.4%2B-informational.svg)
![code style](https://img.shields.io/badge/code%20style-black-000000.svg)

A python library to simplify performing camera calibration and image un-distortion using OpenCV.

## Installation

The camera calibration package can be installed by running:
```bash
pip install git+https://github.com/ukatc/camera-calibration
```

## Usage

The package exposes a set of methods for correcting camera distortion in images and pixel
coordinates, and can also convert pixel coordinates to real world coordinates on the plane of the calibration image.
In all of these, a calibration configuration object, and an indicator of the corrections to perform, must be passed in.

- `camera_calibration.correct_point(point, config, correction_level)` Corrects a single (x, y) point
- `camera_calibration.correct_points(points, config, correction_level)` Corrects an Nx1x2 numpy array of N points
- `camera_calibration.correct_image(image, config, correction_level)` Corrects an OpenCV image

The configuration object is an instance of `camera_calibration.Config`.
It can either have its attributes set:
- through the various `populate_*()` methods on the object, which calculate the properties using a reference grid of
points, or an image that contains them
- directly on the object, if the values are known
- by creating an instance from a saved config using the classes `save()` and `load()`, or `to_dict()` and `from_dict()`
  methods

The correction levels are defined in the `camera_calibration.Correction` enum. It contains three base transforms,
as well as combined values that will perform multiple corrections in a single method call. The base transforms are
- `lens_distortion` Corrects for the camera's lens distortion
- `keystone_distortion` Corrects for any angle offset from the camera and the calibration image's normal. When applied
to the calibration image used, the calibration grid will appear rectangular
- `real_coordinates` Only affects points. Project's the pixel coordinates to a position on the calibration
image's plane based on the grid's corners and size

Each base transform will only work correctly if the prior base transforms have been applied. The combined transforms
apply each component base transform in the correct order.

Example scripts can be found [here](example_scripts).

### In case of distortion in corrected images 

If the images produced by the undistortion methods are wildly distorted, such as all input pixels being compressed into
a small curved sliver across the output image after lens correction, or rotation by 90 degrees after keystone
correction, this may be fixable by swapping the `row` and `column` parameters passed to the configuration methods.

(Based on an OpenCV note [here](https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#calibratecamera))

## Dev environment

The dev environment can be set up by running
```bash
pip install -r requirements.txt
```

### Tests

Tests are performed using [pytest](https://docs.pytest.org/en/latest/).
They can be run from the package's directory with the `pytest` command.

These should be run in environments for all supported versions to ensure compatibility.

### Code style

This package uses [Black](https://black.readthedocs.io/en/stable/index.html) formatting.
It is advisable to configure the formatter to run automatically,
following either its IDE or version control integration guide.

### Versioning

This package uses [semantic versioning](https://semver.org/).

Package version number is stored [here](camera_calibration/__version__.py).
