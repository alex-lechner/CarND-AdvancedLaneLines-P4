# Finding Lane Lines on the Road (Advanced)

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane bound♦aries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/pipeline_test_undistorted.jpg "Undistorted"
[image2]: ./output_images/pipeline_test_masked.jpg "Masked"
[image3]: ./test_images/test5.jpg "Road Test Image"
[image4]: ./output_images/pipeline_test_color_binary_hls.jpg "Binary Example with HLS"
[image5]: ./output_images/pipeline_test_color_binary_combined.jpg "Binary Example"
[image6]: ./output_images/pipeline_test_warped.jpg "Warp Example"
[image7]: ./output_images/pipeline_test_lane_detection.jpg "Fit Visual"
[image8]: ./output_images/pipeline_test_result.jpg "Output"
[image9]: ./output_images/chessboard_calibration2.jpg "Distorted Chessboard"
[image10]: ./output_images/undistorted_calibration2.jpg "Undistroted Chessboard"
[video1]: ./project_video_output.mp4 "Video"

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is located in lines 14 - 40 of the file called `lane_finding.py`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image9] ![alt text][image10]

(The image above is distorted and the image below is undistorted)

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![alt text][image3] ![alt text][image1]

(The image above is distorted and the image below is undistorted)

Afterwards I masked my image to only have the region of interest:

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. For this I have two function called ```combined_threshold()``` (Lines 157 - 171) and ```hls_with_sobelx()``` (Lines 174 - 192).  Here are the examples of my output for this step.

![alt text][image4] ![alt text][image5]

(The image above is the output from ```hls_with_sobelx()``` and the image below is the output from```combined_threshold()```)

As you can see the left image does a pretty good job by detecting the lane lines so that's why I've used ```hls_with_sobelx()``` for the image processing step.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 52 through 67 in the file `lane_finding.py`.  The `warper()` function takes as inputs an image (`img`). I've hardcoded the source and destination points into the function in the following manner:

```python
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
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image6]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did the lane detection by first searching for the lane lines from scratch and afterwards finding them based on the previous detections (Lines 234 - 308 in ```lane_detection()``` and lines 311 -334 in ```lane_detection_from_previous()```). Also I fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image7]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did the curvature and position calculation with the ```calculate_curvature()``` function in lines 345 through 361 in `lane_finding.py` file.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 362 through 375 in the function `draw_lines()` and and unwarped the image in lines 70 through 73 in the function `unwarper()`.  Here is an example of my result on a test image:

![alt text][image8]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

For a short moment my pipeline has the problem to detect the yellow lane line if the pavement matches the color of the lane line. Also it might be likely to fail the lane detection if there are temporary lane lines which look quite the same as the normal ones in a binary image.

The pipeline could be more robust if I were playing around a little bit more with different thresholds and other color spaces e.g.: HSV and combining them into one binary output image. The pipeline reacts very sensitive when playing with different kernel sizes and thresholds so fine tuning these parameters might be a potential improvement.

I think it's very important how well the lane detection works based on the daytime because the detection might not work very well if the sun light is shining right into the camera. 
