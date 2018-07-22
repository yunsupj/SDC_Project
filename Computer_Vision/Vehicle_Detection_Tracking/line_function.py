import numpy as np
import cv2
import glob
import matplotlib.image as mpimg

def calibration():
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('./camera_cal/calibration*.jpg')

    # Step through the list and search for chessboard corners
    for i in range(len(images)):
        img = cv2.imread(images[i])
        y, x = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Calibrate Images and undistortion with distortion coefficients
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (x, y), None, None)
    return mtx, dist

def undistortion(img, mtx, dist):
    '''
    This function makes undistortion image (2D plane image) from 3D image in real world
    INPUT:
        - img: image matrix or image file in 3D (distorted image)
        - mtx: Camera calibration matrix from chessboard images
        - dist: Distortion Coefficients
        
    OUTPUT:
        - 2D plane image (no distortion)
    '''
    if type(img) == str:
        img = cv2.imread(img)
    elif type(img) == type(np.array([0])):
        img = img
    else:
        print('Check the Image Files...') 
    
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

def tran_perspective(img, mtx, dist):
    '''
    This function makes perspective transformed image (bird eyes) with source points and destination points.
    The perspective changes into bird eyes images from the trapezoid perspective on the original image
      - Source Points: Trapezoid perspective from 4 points on the original images
      - Destination Points: 4 destination points, source points are tranformed into destination points location
      
    INPUT:
        - img: image matrix or image file in 3D (distorted image)
        - mtx: Camera calibration matrix from chessboard images
        - dist: Distortion Coefficients
        
    OUTPUT:
        - bird eyes images
    '''
    
    img = undistortion(img, mtx, dist)
    y, x = img.shape[:2]                       # row, column == y, x

    # Source points
    sp1 = [0, y-10]                      # Left Bottom
    sp2 = [x*.43, y*.64]                 # Left Top
    sp3 = [x*.57, y*.64]                 # Right Top
    sp4 = [x, y-10]                      # Right Bottom
    
    # Destination points
    dp1 = [0, y]                         # Left Bottom
    dp2 = [0, 0]                         # Left Top
    dp3 = [x, 0]                         # Right Top
    dp4 = [x, y]                         # Right Bottom

    img_line = img.copy()
    points = np.array([sp3, sp4, sp1, sp2], np.int32)
    cv2.polylines(img_line, [points], 1, (255, 0, 255), thickness=3)
    
    source = np.float32([sp3, sp4, sp1, sp2])
    destination = np.float32([dp3, dp4, dp1, dp2])
    
    M = cv2.getPerspectiveTransform(source, destination)
    M_inv = cv2.getPerspectiveTransform(destination, source)
    warped = cv2.warpPerspective(img, M, (x, y))
    
    return img_line, warped, M_inv

def ls_channel_thresh(img, l_thresh, s_thresh):
    '''
    Apply binary Threshold in S-Channel and L-Channel after converted into HLS color space
    S_Thresh: It detects yellow and white lanes
    L_Thresh: It helps to avoid shadowed and darked pixels
    '''
    light = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:,:,1]
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:,:,2]
    
    combine_binary = np.zeros_like(sat)

    sat_binary = (sat > s_thresh) & (sat <= 255)
    light_binary = (light > l_thresh) & (light <= 255)
    
    ls_thresh = (light_binary & sat_binary)
    combine_binary[ls_thresh] = 1
    
    return combine_binary, ls_thresh

def sobelx_thresh(img, thresh_min, thresh_max):
    '''
    Apply the gradient(sobel) Threshold on the horizontal gradient.
    Sobel(Gradient) : Apply Horizontal Gradient
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)             #convert to gray scale
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)              #apply gradient threshold on the horizontal gradient
    
    abs_sobelx = np.abs(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx / np.max(abs_sobelx))
    sobel_binary = np.zeros_like(scaled_sobel)
    
    sobel_thresh = (scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)
    sobel_binary[sobel_thresh] = 1
    return sobel_binary

def direction_thresh(img, thresh_min, thresh_max):
    '''
    Apply the gradient direction Threshold to find the direction.
    Direction threshold : Apply gradient direction threshold so that only edges closer to vertical are detected.
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)             #convert to gray scale
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)      #apply gradient threshold on the horizontal gradient
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)      #apply gradient threshold on the vertical gradient

    abs_sobelx = np.abs(sobelx)
    abs_sobely = np.abs(sobely)
        
    #Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    direction = np.abs(np.arctan2(abs_sobely, abs_sobelx))
    
    direction_binary = np.zeros_like(direction)
    direction_thresh = (direction >= thresh_min) & (direction <= thresh_max)
    direction_binary[direction_thresh] = 1
    return direction_binary

def measure_curvature(img, x_pix):
    ym_per_pix = 30/720         # meters per pixel in y dimension
    xm_per_pix = 3.7/700        # meters per pixel in x dimension

    # Define maximum y-value corresponding to the bottom of the image, If no pixels were found return None
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    y_eval = np.max(ploty)

    # Fit new polynomials to x, y in world space
    fit_cur = np.polyfit(ploty*ym_per_pix, x_pix*xm_per_pix, 2)
    curverad = ((1 + (2*fit_cur[0]*y_eval*ym_per_pix + fit_cur[1])**2)**1.5) / np.absolute(2*fit_cur[0])
    return curverad
