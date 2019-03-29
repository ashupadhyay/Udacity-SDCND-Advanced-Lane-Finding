## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./files/warped.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./files/final.png "Output"
[video1]: ./project_video.mp4 "Video"
[image7]: ./files/lane_dist_undist.png "lane_undistorted"
[image8]: ./files/sobelx.png
[image9]: ./files/sobely.png
[image10]: ./files/sobel_combined.png
[image11]: ./files/binary.png
[image12]: ./files/rgb.png
[image13]: ./files/hls.png
[image14]: ./files/color_pipeline.png

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  
My writeup is a markdown file you can find it [here](https://github.com/ashupadhyay/Udacity-SDCND-Advanced-Lane-Finding/blob/master/writeup_template.md) 

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the Seventh to the Eleventh code cell of the IPython notebook located in "Advanced Lane finding.ipynb" 

Two helpful functions provided by the OpenCV are `cv2.findChessboardCorners` and  `cv2.drawChessboardCorners`.
The important steps to calibrate your camera is:
1. Collect the object points(internal indices points) from multiple images of a chessboard taken from different angles with the same camera. 
2. Now also, we would need the internal image points to feed to the function which would in turn calibrate the camera.

So, the process is, we find the internal image points using `cv2.findChessboardCorners` that we feed to the function `cv2.calibrateCamera`, this returns to us `cameraMatrix` and `distortion coefficients`. These can then be used by the OpenCV `undistort` function to undo the effects of distortion on any image produced by the same camera.

Note: 
1. There is an assumption that the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  
2. Some of the chessboard images don't appear because findChessboardCorners was unable to detect the desired number of internal corners. For them, the function `image_dist_undist` print appropriate messages.

`image_dist_undist` is the name of the function which receives the input file that we need to undistort, it undistorts the image and plots it (undistorted and distorted, side by side)!

![alt text][image1]


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

I simply used the function `findChessboardCorners` to get the chessboard corners. After I received the corners I fed it to the `calibrateCamera` functions which returned me the `cameraMatrix` and `distortion coefficients` which were used by the function `undistort` to correct the distortion of the image. 
After applying the above steps, our camera is calibrated and we can use any image to be undistorted. I tried appying it on our test images and here is the final output of the image:
![alt text][image7]

When we look through the lane lines, we might not be able to observe the effect of `undistort` but when we look at the hood of the car, we can observe the change in the distortion of the image.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I have marked every code required with appropriate headings above the code section. I have appropriately described with much details in ipython notebook everything needed. I am summarizing below.

I tried to understand how individually applying sobelx and sobely operators on image would look like,

Sobelx
![alt text][image8]

Sobely
![alt text][image9]

### Using combined sobelx and sobely 

This uses the square root of the combined squares of Sobelx and Sobely, which check for horizontal and vertical gradients (shifts in color, or in the case of our images, lighter vs. darker gray after conversion), respectively.This uses the square root of the combined squares of Sobelx and Sobely, which check for horizontal and vertical gradients (shifts in color, or in the case of our images, lighter vs. darker gray after conversion), respectively.

```
def mag_thresh(img, sobel_kernel=9, mag_thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude 
    grad_mag = np.sqrt(sobelx**2+sobely**2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scale_factor = np.max(grad_mag)/255 
    grad_mag = (grad_mag/scale_factor).astype(np.uint8) 
    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(grad_mag)
    binary_output[(grad_mag >= mag_thresh[0]) & (grad_mag <= mag_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output
mag_binary = mag_thresh(image, sobel_kernel=3, mag_thresh=(30, 100))
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(mag_binary, cmap='gray')
ax2.set_title('Thresholded Magnitude', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
```

![alt text][image10]

# Direction of the Gradient
Gradient magnitude is at the heart of Canny edge detection, and is why Canny works well for picking up all edges.<br>
In the case of lane lines, we're interested only in edges of a particular orientation. So now we will explore the direction, or orientation, of the gradient.

The direction of the gradient is simply the inverse tangent (arctangent) of the yy gradient divided by the xx gradient
$$arctan(sobel_x/sobel_y)$$

```
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    # 5) Create a binary mask where direction thresholds are met
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output
```

Tried different combinations here and explored wonderful variations.
Now I am trying to combine all the conditions here as a selection for pixels where both the `x` and `y` gradients meet the threshold criteria, or the gradient magnitude and direction are both within their threshold values.

```
ksize = 3 # Choose a larger odd number to smooth gradient measurements
def combined_transform(image):
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20,100))
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(np.pi/6, np.pi/2))
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    return combined
```
Final output looks like this
![alt text][image11]


Any work done above finally didn't give us something substantial even after so many operations.
Because the lane lines could be of any color and what would happen if we want to clearly identify them?
All above operations could not result in something we can use, so tried to use different available color spaces and color channels.


### RGB color thresolds and HLS color thresholds

I tried to look at the different channels an image has,
Our lane line has yellow color, and Yellow is essentially made up of red and green colors, we can see that red has moore number of visible pixels as compared to blue.

For RGB color thresholds the output was,
![alt text][image12]

For HLS color thresholds, the output was,
![alt text][image13]

As it is visible here that the S channel gave the most clear output for out lane lines, I selected it for further transformations.

Then I used Sobelx, L channel and S channel to make a combined image and created a color pipeline It's output is here
![alt text][image14]


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `transform_image(image)`, which is properly mentioned in the notebook with required explanation. The `transform_image()` function takes as inputs an image (`img`). Inside this function I define source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 470      | 200, 0        | 
| 717, 470      | 1000,0        |
| 260, 680      | 200,680       |
| 1043,680      | 1000,680      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I have an image which has been perspective transformed, binary image. The next step is to plot a histogram based on where the binary activations occur across the x-axis, as the high points in a histogram are the most likely locations of the lane lines.
This can be achieved using the following code
```histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)```

From this histogram we would find the base points of left and right lanes 
```
midpoint = np.int(histogram.shape[0]/2)
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint
```

Now we would use the sliding window method to select a block of image and then try to fit the identified pixels on a polynomial line.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Radius of curvature is calculated in my code in the function `radius_curvature`

```

ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    y_eval = np.max(ploty)
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    
    # Calculate the new radii of curvature
    left_curvature =  ((1 + (2*left_fit_cr[0] *y_eval*ym_per_pix + left_fit_cr[1])**2) **1.5) / np.absolute(2*left_fit_cr[0])
    right_curvature = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    # Calculate vehicle center
    #left_lane and right lane bottom in pixels
    left_lane_bottom = (left_fit[0]*y_eval)**2 + left_fit[0]*y_eval + left_fit[2]
    right_lane_bottom = (right_fit[0]*y_eval)**2 + right_fit[0]*y_eval + right_fit[2]
    
    # Lane center as mid of left and right lane bottom                        
    lane_center = (left_lane_bottom + right_lane_bottom)/2.
    center_image = 640
    center = (lane_center - center_image)*xm_per_pix #Convert to meters
    position = "left" if center < 0 else "right"
    center = "Vehicle is {:.2f}m {}".format(center, position)
    
    # Now our radius of curvature is in meters
    return left_curvature, right_curvature, center 

```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the code section draw_on_image(). Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://github.com/ashupadhyay/Udacity-SDCND-Advanced-Lane-Finding/blob/master/files/project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

I think mainly the fluctuations are seen on the road when either the color of the road becomes too much white or the car passes under a shadow region.

I am really unsure at this moment how can I proceed, I would say that I would adjust the thresholds a bit more and would see how it turns out.
