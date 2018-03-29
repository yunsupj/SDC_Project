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
This is kinds of pre-process for edges detection. The matrix of RGB color image sometimes too complicate to pulling out edges, so this process makes easier(simple) to catch significant gradients from the image matrix. This precesses using openCV package to convert grayscale and using the Gaussian filter([Gaussian Blur](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html)).

![Result: Grayscale][image1]     ![Result: Gaussian Blur][image1]

**Step 2:** Gaussian Blur -> Canny Edge Detection
This step is drawing edges on the image with [Canny Edge Detection](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html). It is possible to use with RGB image matrix, but it prefer to use with grayscaled image matrix because Canny Edge Detection use gradient values(RGB has a lot of noise).

![Result: Canny Edge Detection][image1]

**Step 3:** Canny Edge Detection -> Masked into Image from Canny Edge Detection


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ...

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...