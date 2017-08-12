import glob

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle

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
        image_name = filename.split("\\")[-1]

        # find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # if corners are found, add object points and image points
        if ret:
            imgpoints.append(corners)
            objpoints.append(objp)

            chessboard_image = cv2.drawChessboardCorners(image, (nx, ny), corners, ret)
            mpimg.imsave("output_images/chessboard_" + image_name, chessboard_image)

            undistorted_image = cal_undistort(image, objpoints, imgpoints)
            mpimg.imsave("output_images/undistorted_" + image_name, undistorted_image)

            warped_image = warper(undistorted_image)
            mpimg.imsave("output_images/warped_" + image_name, warped_image)


def cal_undistort(img, object_points, image_points):
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, img_size, None, None)
    calibration = {"mtx": mtx, "dist": dist}
    pickle.dump(calibration, open("calibration.p", "wb"))
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


def warper(img):
    img_size = (img.shape[1], img.shape[0])
    src = np.float32(
        [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
         [((img_size[0] / 6) - 10), img_size[1]],
         [(img_size[0] * 5 / 6) + 60, img_size[1]],
         [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
    dst = np.float32(
        [[(img_size[0] / 4), 0],
         [(img_size[0] / 4), img_size[1]],
         [(img_size[0] * 3 / 4), img_size[1]],
         [(img_size[0] * 3 / 4), 0]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)
    return warped, M, Minv


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient.lower() == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient.lower() == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    else:
        raise ValueError('Error: Please insert x or y for orientation')
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    grad_binary = np.zeros_like(scaled_sobel)
    # 6) Return this mask as your binary_output image
    grad_binary[(scaled_sobel > thresh[0]) & (scaled_sobel < thresh[1])] = 1
    return grad_binary


def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    # 5) Create a binary mask where mag thresholds are met
    mag_binary = np.zeros_like(gradmag)
    # 6) Return this mask as your binary_output image
    mag_binary[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1
    return mag_binary


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    gradient_dir = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    dir_binary = np.zeros_like(gradient_dir)
    # 6) Return this mask as your binary_output image
    dir_binary[(gradient_dir >= thresh[0]) & (gradient_dir <= thresh[1])] = 1
    return dir_binary


def hls_threshold(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    S = hls[:, :, 2]
    hls_binary = np.zeros_like(S)
    # 3) Return a binary image of threshold result
    hls_binary[(S > thresh[0]) & (S <= thresh[1])] = 1
    return hls_binary




# Choose a Sobel kernel size
ksize = 3

# Apply each of the thresholding functions
# gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(0, 255))
# grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(0, 255))
# mag_binary = mag_thresh(image, sobel_kernel=ksize, thresh=(0, 255))
# dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0, np.pi / 2))
# hls_binary = hls_threshold(image, thresh=(0, 255))
#
# combined = np.zeros_like(dir_binary)
# combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
#
# color_binary = np.dstack((np.zeros_like(combined), combined, hls_binary))


# TODO Finish up the pipeline
# def pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
#     img = np.copy(img)
#     # Convert to HSV color space and separate the V channel
#     hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
#     l_channel = hsv[:, :, 1]
#     s_channel = hsv[:, :, 2]
#     # Sobel x
#     sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
#     abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
#     scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
#
#     # Threshold x gradient
#     sxbinary = np.zeros_like(scaled_sobel)
#     sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
#
#     # Threshold color channel
#     s_binary = np.zeros_like(s_channel)
#     s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
#     # Stack each channel
#     # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
#     # be beneficial to replace this channel with something else.
#     color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))
#     return color_binary
#
#
# result = pipeline(image)

def region_of_interest(img):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # Create region mask
    height = img.shape[0]
    width = img.shape[1]
    # Based on the angle of the camera we can assume that at least 50 - 60% from the upper half of the image
    # is not necessary for lane detection
    upper_half = height * .6
    ratio = 4 / 7
    vertices = np.array([
        [(20, height),
         ((1 - ratio) * width, upper_half),
         (ratio * width, upper_half),
         (width - 20, height)]
    ], dtype=np.int32)

    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


# camera_calibration("camera_cal/calibration*.jpg")
# undistorted = cal_undistort(images, objpoints, imgpoints)

test_image = mpimg.imread('test_images/test5.jpg')

img_masked = region_of_interest(test_image)

img_bird_view, M, Minv = warper(img_masked)

plt.imshow(img_masked)
plt.show()
plt.imshow(img_bird_view)
plt.show()

# load pickle data
calib_pickle = pickle.load(open('calibration.p', 'rb'))
mtx = calib_pickle['mtx']
dist = calib_pickle['dist']

# TODO define function for lane detection

# TODO test on video stream
