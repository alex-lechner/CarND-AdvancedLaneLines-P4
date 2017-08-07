import glob

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

# Arrays to store object points and image points from all images
# objpoints = 3D points in real world space; imgpoints = 2D points in image plane;
objpoints, imgpoints = [], []


def camera_calibration(folder, nx=9, ny=6):
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    for filename in glob.glob(folder):
        # read in image as RGB and convert to gray scale
        image = mpimg.imread(filename)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # if corners are found, add object points and image points
        if ret:
            imgpoints.append(corners)
            objpoints.append(objp)

            image = cal_undistort(image, objpoints, imgpoints)
            # image = cv2.drawChessboardCorners(image, (nx, ny), corners, ret)
            plt.imshow(image)
            plt.show()


def cal_undistort(img, object_points, image_points):
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, img_size, None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


camera_calibration("camera_cal/calibration*.jpg")

# undistorted = cal_undistort(images, objpoints, imgpoints)
