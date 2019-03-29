# Sample Images

A collection of calibration images used by the package's tests and example scripts.
All of them are taken by the same camera, so should give fairly similar lens distortion configurations.

- **002.bmp** - An image of chessboard calibration image that results in an 8x6 grid that is 90.06mm x 64.45mm.
Points for the grid are determined by the corner intersections of black squares.
- **002h.bmp** - A high contrast and brightness copy of 002.bmp.
This makes the pattern detectable by OpenCV without having to perform a thresholding step first.
- **distcor_01_cleaned.bmp** - An image of a high precision dot grid with black border removed and dirt edited out so
that OpenCV can detect all the points in the grid with an appropriate blob detector. The grid is 170 dots by 116 dots,
with a 0.5mm spacing between the centers of each row and column.
- **distcor_04_cleaned.bmp** - Another image of the same high precision dot grid as distcor_01_cleaned.bmp.
However, this image has been used to produce more accurate calibration configurations.
- **mocked checkboard.png** - A chessboard made by drawing quadrilaterals over distcor_01_cleaned.bmp.
This gives a 15x10 grid that measures 70mm x 45mm.
