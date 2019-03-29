"""
Generate a calibration configuration using a chessboard image.
Then use that configuration to generate distortion corrected image files of that chessboard.
"""
import camera_calibration as calib
import cv2

image_path = "../sample_images/002h.bmp"
rows = 6
cols = 8

config = calib.Config()
config.populate_lens_parameters_from_chessboard(image_path, rows, cols)
config.populate_keystone_and_real_parameters_from_chessboard(
    image_path, cols, rows, 90.06, 64.45
)

bgr = cv2.imread(image_path)
undistorted = calib.correct_image(bgr, config, calib.Correction.lens_distortion)
cv2.imwrite("undistorted.png", undistorted)
grid_aligned = calib.correct_image(
    undistorted, config, calib.Correction.keystone_distortion
)
cv2.imwrite("grid_aligned.png", grid_aligned)
