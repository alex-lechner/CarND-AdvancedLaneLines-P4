import glob
import pickle
from pathlib import Path

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoFileClip

from Line import Line


def camera_calibration(folder, nx=9, ny=6):
    # Arrays to store object points and image points from all images
    # objpoints = 3D points in real world space; imgpoints = 2D points in image plane;
    objpoints, imgpoints = [], []

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

        undistorted_image = undistort_calibration(image, objpoints, imgpoints)
        mpimg.imsave("output_images/undistorted_" + image_name, undistorted_image)


def undistort_calibration(img, object_points, image_points):
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


def unwarper(img, Minv):
    img_size = (img.shape[1], img.shape[0])
    unwarped = cv2.warpPerspective(img, Minv, img_size, flags=cv2.INTER_NEAREST)
    return unwarped


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255), convert_gray=False):
    # Apply the following steps to img
    # 1) Convert to grayscale if convert_gray is True
    if convert_gray:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient.lower() == 'x':
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient.lower() == 'y':
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    else:
        raise ValueError('Error: Please insert \'x\' or \'y\' for orientation')
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


def mag_thresh(img, sobel_kernel=3, thresh=(0, 255), convert_gray=False):
    # 1) Convert to grayscale if convert_gray is True
    if convert_gray:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
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


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2), convert_gray=False):
    # 1) Convert to grayscale if convert_gray is True
    if convert_gray:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
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


def hls_threshold(img, channel='s', thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # 2) Select a channel
    if channel.lower() == 'h':
        color_channel = hls[:, :, 0]
    elif channel.lower() == 'l':
        color_channel = hls[:, :, 1]
    elif channel.lower() == 's':
        color_channel = hls[:, :, 2]
    else:
        raise ValueError('Error: Please insert \'h\', \'l\' or \'s\' for channel')
    hls_binary = np.zeros_like(color_channel)
    # 3) Apply a threshold to the S channel
    hls_binary[(color_channel > thresh[0]) & (color_channel <= thresh[1])] = 1
    # 4) Return a binary image of threshold result
    return hls_binary


def combined_threshold(img, kernel_size=3):
    # Apply each of the thresholding functions
    # l_channel = hls_threshold(img, channel='s', thresh=(10, 100))
    # plt.imshow(l_channel, cmap='gray')
    # plt.show()
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=kernel_size, thresh=(20, 100), convert_gray=True)
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=kernel_size, thresh=(20, 100), convert_gray=True)
    s_channel = hls_threshold(img, thresh=(10, 230))
    mag_binary = mag_thresh(s_channel, sobel_kernel=kernel_size, thresh=(30, 180))
    dir_binary = dir_threshold(s_channel, sobel_kernel=9, thresh=(0.7, 1.4))

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    return combined


def hls_with_sobelx(img, kernel_size=3, s_thresh=(170, 250), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    # Sobel x
    sxbinary = abs_sobel_thresh(l_channel, orient='x', sobel_kernel=kernel_size, thresh=sx_thresh)

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    # color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))
    combined = np.zeros_like(s_channel)
    combined[(sxbinary == 1) | (s_binary == 1)] = 1
    return combined


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


def lane_detection(binary_warped, nwindows=9):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds, right_lane_inds = [], []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
            nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
            nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            left_lane.detected = True
        else:
            left_lane.detected = False

        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            right_lane.detected = True
        else:
            right_lane.detected = False

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    return left_fit, right_fit


def lane_detection_from_previous(binary_warped):
    left_fit, right_fit = left_lane.current_fit, right_lane.current_fit
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
        nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
    right_lane_inds = (
        (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
            nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    return left_fit, right_fit


def generate_values(binary_warped, left_fit, right_fit):
    # Generate x and y values
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    leftx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    rightx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    return ploty, leftx, rightx


def calculate_curvature(ploty, leftx, rightx):
    y_eval = np.max(ploty)
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    offset = ((leftx[-1] + rightx[-1]) / 2 - 640) * xm_per_pix
    return left_curverad, right_curverad, offset


def draw_lines(warped_img, leftx, rightx, ploty):
    warp_zero = np.zeros_like(warped_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([leftx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([rightx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    return color_warp


def video_pipeline(img, ksize=5, nwindows=9, save=False):
    # Check if calibration picke file exists and if not calibrate the camera
    if not Path("calibration.p").is_file():
        # Calibrate Camera
        camera_calibration("camera_cal/calibration*.jpg")

    calib_pickle = pickle.load(open('calibration.p', 'rb'))
    mtx = calib_pickle['mtx']
    dist = calib_pickle['dist']

    # undistort the input image
    undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)
    if save:
        mpimg.imsave("output_images/pipeline_test_undistorted.jpg", undistorted_img)

    # mask the undistorted image
    masked_img = region_of_interest(undistorted_img)
    if save:
        mpimg.imsave("output_images/pipeline_test_masked.jpg", masked_img)

    # convert image to a colored binary image (hls version)
    color_binary_hls_img = hls_with_sobelx(masked_img, kernel_size=ksize)
    if save:
        mpimg.imsave("output_images/pipeline_test_color_binary_hls.jpg", color_binary_hls_img, cmap='gray')

    # convert image to a colored binary image (combined dir, mag and sobel)
    color_binary_combined_img = combined_threshold(masked_img, kernel_size=ksize)
    if save:
        mpimg.imsave("output_images/pipeline_test_color_binary_combined.jpg", color_binary_combined_img, cmap='gray')

    # warp image to birds-eye view
    warped_img, M, Minv = warper(color_binary_hls_img)
    if save:
        mpimg.imsave("output_images/pipeline_test_warped.jpg", warped_img, cmap='gray')

    # detect the lane lines
    if left_lane.detected or right_lane.detected:
        left_fit, right_fit = lane_detection_from_previous(warped_img)
    else:
        left_fit, right_fit = lane_detection(warped_img, nwindows=nwindows)

    left_lane.current_fit, right_lane.current_fit = left_fit, right_fit

    # generate x and y values
    ploty, leftx, rightx = generate_values(warped_img, left_fit, right_fit)
    if save:
        plt.axis('off')
        plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off',
                        labelright='off', labelbottom='off')
        plt.imshow(warped_img, cmap='gray')
        plt.plot(leftx, ploty, color='yellow')
        plt.plot(rightx, ploty, color='yellow')
        plt.savefig("output_images/pipeline_test_lane_detection.jpg", bbox_inches='tight', pad_inches=0)
        plt.close()

    left_lane.allx, right_lane.allx = leftx, rightx
    left_lane.ally = right_lane.ally = ploty

    # calculate the curvature
    left_curvrad, right_curvrad, offset = calculate_curvature(ploty, leftx, rightx)
    left_lane.radius_of_curvature, right_lane.radius_of_curvature = left_curvrad, right_curvrad

    # Create an image to draw the lines on
    color_warp = draw_lines(warped_img, leftx, rightx, ploty)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    unwarped_img = unwarper(color_warp, Minv)
    # Combine the result with the original image
    result_img = cv2.addWeighted(undistorted_img, 1, unwarped_img, 0.3, 0)
    if save:
        mpimg.imsave("output_images/pipeline_test_result.jpg", result_img)
    else:
        text = "Left curvature: {:.2f}m\n".format(left_curvrad)
        text += "Right curvature: {:.2f}m\n".format(right_curvrad)
        if offset > 0:
            text += "Distance from left: {:.2f}m".format(offset)
        elif offset < 0:
            text += "Distance from right: {:.2f}m".format(offset)
        else:
            text += "Vehicle is at the center between lane lines."
        # Hack because OpenCV doesN#t support \n character in putText()
        y0, dy = 50, 48
        for i, line in enumerate(text.split('\n')):
            y = y0 + i * dy
            cv2.putText(result_img, line, (50, y), cv2.FONT_HERSHEY_COMPLEX, 1, [255, 255, 255])

    return result_img


# Choose a Sobel kernel size and the number of sliding windows
ksize = 5
nwindows = 9

# load test image
test_image = mpimg.imread('test_images/test5.jpg')
left_lane = Line()
right_lane = Line()
result = video_pipeline(test_image, ksize=ksize, nwindows=nwindows, save=True)

video_output_file = 'project_video_output.mp4'
video_input_file = VideoFileClip("./project_video.mp4")
video_output = video_input_file.fl_image(video_pipeline)
video_output.write_videofile(video_output_file, audio=False)
