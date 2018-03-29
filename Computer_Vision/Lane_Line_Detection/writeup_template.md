# **Finding Lane Lines on the Road**

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

---

**Finding Lane Lines on the Road**

The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle.  Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm.

The goals of this project to detect lane lines in images using Python and [OpenCV](https://opencv.org/).  
* OpenCV means “Open-Source Computer Vision”, which is a package that has many useful tools for analyzing images.

<img src=“###outpufile###” width=“480” alt=“Combined Image” />

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.
### 1. The steps of this project are the following:

#### My pipeline consisted of 5 steps. 

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...

If you’d like to include images to show how the pipeline works, here is how to include an image: 

**Step 1:** Image -> Grayscale -> Gaussian Blur
This is kinds of pre-process for edges detection. The matrix of RGB color image sometimes too complicate to factor out edges, so this process makes easier(simple) to catch significant gradients from the image matrix. This precesses using openCV package to convert grayscale and using Gaussian filter([Gaussian Blur](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html)).

![alt text](https://github.com/yunsupj/SDC_Project/blob/master/Computer_Vision/Lane_Line_Detection/test_images/output_gray.jpg?raw=true "Result: Grayscale")

**Step 2:** Gaussian Blur -> Canny Edge Detection
This step is drawing edges on the image with [Canny Edge Detection](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html). It is possible to use with RGB image matrix, but it prefer to use with grayscaled image matrix because Canny Edge Detection use gradient values(RGB has a lot of noise).

![alt text](https://github.com/yunsupj/SDC_Project/blob/master/Computer_Vision/Lane_Line_Detection/test_images/output_edges.jpg?raw=true "Result: Canny Edge Detection")

**Step 3:** Canny Edge Detection + Mask(Region of Interest) Image -> Masked Canny Edge Detection
This step is cleaning noised edges. The result of the Canny edge detection is edges of lane-line and other edges which are not needed. The first, set the four points, and draw four straight line which passes through points in order to make parallelogram region(you can make different shapes of the region with different number of points). It is called Region of Interest(Mask). The second, making an image of black and white; White = inside mask(Region of Interest), Black = outside mask(Region of Interest). The final, overlapping of two images; Canny Edge Detection + Mask(Region of Interest) Image -> Masked Canny Edge Detection.
* openCV package used for making line([cv2.fillpoly()](https://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html) and overlapping images([cv2.bitwise_and()](https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html). 

![alt text](https://github.com/yunsupj/SDC_Project/blob/master/Computer_Vision/Lane_Line_Detection/test_images/output_mask.jpg?raw=true "Result: Mask(Region of Interest)")

![alt text](https://github.com/yunsupj/SDC_Project/blob/master/Computer_Vision/Lane_Line_Detection/test_images/output_mask_img.jpg?raw=true "Result: Mask(Region of Interest) + Canny Edge Detection")

**Step 4:** Masked Canny Edge Detection -> Detect Lane Line
This step is detecting lane line with using [Hough Line Transform](https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/hough_lines/hough_lines.html). The Hough line transformation is making lines into hough space based on two points from the original image(in this case, Image of Masked Canny Edge Detection). openCV package provides this process as [cv2.HoughLinesP()](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html). The single output of HoughLinesP() is x and y values of two points, and making lines with these values. 

![alt text](https://github.com/yunsupj/SDC_Project/blob/master/Computer_Vision/Lane_Line_Detection/test_images/output_lines.jpg?raw=true "Result: Detect Lane Line")

**Step 5:** Complete Detect Lane Line: Make one line from segmented lines




### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ...

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...