# **Finding Lane Lines on the Road**

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

---

**Finding Lane Lines on the Road**

The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle.  Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm.

The goals of this project to detect lane lines in images using Python and [OpenCV](https://opencv.org/). OpenCV means “Open-Source Computer Vision”, which is a package that has many useful tools for analyzing images.

<img src="https://github.com/yunsupj/SDC_Project/blob/master/Computer_Vision/Lane_Line_Detection/test_images/output_solidYellowLeft.jpg" width="480" alt="Combined Image" />

---

### 1. The steps of this project are the following:

### My pipeline consisted of 5 steps. 

**Step 1:** Image -> Grayscale -> Gaussian Blur

This is kinds of pre-process for edges detection. <br />The matrix of RGB color image sometimes too complicate to factor out edges, so this process makes easier(simple) to catch significant gradients from the image matrix. 
This precesses using openCV package to convert grayscale and using Gaussian filter([Gaussian Blur](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html)).

##### Result: Grayscale
<img src="https://github.com/yunsupj/SDC_Project/blob/master/Computer_Vision/Lane_Line_Detection/test_images/output_gray.jpg" width="480" alt="Combined Image" /> 
<br />
<br />

**Step 2:** Gaussian Blur -> Canny Edge Detection

This step is drawing edges on the image with [Canny Edge Detection](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html). <br />
It is possible to use with RGB image matrix, but it prefer to use with grayscaled image matrix because Canny Edge Detection use gradient values (RGB has a lot of noise).

##### Result: Canny Edge Detection
<img src="https://github.com/yunsupj/SDC_Project/blob/master/Computer_Vision/Lane_Line_Detection/test_images/output_edges.jpg" width="480" alt="Combined Image" /> 
<br />
<br />

**Step 3:** Canny Edge Detection + Mask (Region of Interest) Image -> Masked Canny Edge Detection

This step is cleaning noised edges. The result of the Canny edge detection is edges of lane-line and other edges which are not needed. 
> The first, set the four points and draw four straight line which passes through points in order to make parallelogram region (you can make different shapes of the region with different number of points). It is called Region of Interest (Mask). <br />
The second, making an image of black and white; White = inside mask (Region of Interest), Black = outside mask (Region of Interest). <br /> The final, overlapping of two images; Canny Edge Detection + Mask (Region of Interest) Image -> Masked Canny Edge Detection.

* openCV package used for making line, [cv2.fillpoly()](https://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html), and overlapping images, [cv2.bitwise_and()](https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html). 

##### Result: Mask &nbsp Result: Mask + Canny Edge Detection
<img src="https://github.com/yunsupj/SDC_Project/blob/master/Computer_Vision/Lane_Line_Detection/test_images/output_mask.jpg" width="360" alt="Combined Image" />     <img src="https://github.com/yunsupj/SDC_Project/blob/master/Computer_Vision/Lane_Line_Detection/test_images/output_mask_img.jpg" width="360" alt="Combined Image" /> 
<br />
<br />

**Step 4:** Masked Canny Edge Detection -> Detect Lane Line

This step is detecting lane line with using [Hough Line Transform](https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/hough_lines/hough_lines.html).<br /> The Hough line transformation is making lines into hough space based on two points from the original image (in this case, Image of Masked Canny Edge Detection). openCV package provides this process as [cv2.HoughLinesP()](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html). <br />The single output of HoughLinesP() is x and y values of two points, and making lines with these values. 

##### Result: Detect Lane Line
<img src="https://github.com/yunsupj/SDC_Project/blob/master/Computer_Vision/Lane_Line_Detection/test_images/output_lines.jpg" width="480" alt="Combined Image" /> 
<br />
<br />

**Step 5:** Complete Detect Lane Line: Make one line from segmented lines

This step is making a single Left and Right line from segmented lines. <br />The original draw_line() function draws many segmented lines on left and right side. In order to make a single Left and Right line, it needs to calculate average of slope and center of coordinate,(x, y) value, for Left and Right.<br /> The result of new_draw_line() function is new (x1, y1, x2, y2) values which can a single Left and Right line with [cv2.line()](https://docs.opencv.org/trunk/dc/da5/tutorial_py_drawing_functions.html).

##### Result: Detect Lane Line - Single Line
<img src="https://github.com/yunsupj/SDC_Project/blob/master/Computer_Vision/Lane_Line_Detection/test_images/output_solidYellowLeft.jpg" width="480" alt="Combined Image" /> 
<br />
<br />

#### Result: solidWhiteRight
<a href="https://imgflip.com/gif/27ewij"><img src="https://i.imgflip.com/27ewij.gif" title="made at imgflip.com"/></a>
<br />
<br />

#### Result: solidYellowLeft
<a href="https://imgflip.com/gif/27ewmm"><img src="https://i.imgflip.com/27ewmm.gif" title="made at imgflip.com"/></a>
<br />
<br />

### 2. Identify potential shortcomings with your current pipeline
For challenge video, there is some error because NaN values exist after filter based on arctangent values.<br />
Also, this project designed only for detecting straight lane line, so this is not really work in real roads.

### 3. Suggest possible improvements to your pipeline
This is the very first step of computer vision for the self-driving car. My pipeline is OK for normal condition of straight lane line, but in other cases, it might have many problems even if I can figure it out filter issues.
