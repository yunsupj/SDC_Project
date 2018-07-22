import numpy as np
import cv2
import glob
from line_function import *
import matplotlib.image as mpimg

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self, mtx, dist):
        # was the line detected in the last iteration?
        self.detected = False
        self.left_fit = None
        self.right_fit = None
        self.ploty = None
        
        # use mtx and dist values from chessboard images
        # * mtx: Camera calibration matrix from chessboard images
        # * dist: Distortion Coefficients
        self.mtx = mtx
        self.dist = dist

    def fit_initial(self, all_combine_binary):
        '''
        This function makes the first poly fit lines on the left and right.
        It uses a sliding windows based on pixel peaks on left and right lane line image.
        '''
        n_windows = 10         # Number of window size
        margin = 50            # Set the width of the windows +/- margin
        minpix = 100            # Set minimum number of pixels found to recenter window

        out_img = np.dstack((all_combine_binary, all_combine_binary, all_combine_binary))*255
        
        # Set height of windows, 720/n_windows(10) = 72
        window_height = np.int(all_combine_binary.shape[0]/n_windows)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = all_combine_binary.nonzero()
        nonzero_y = nonzero[0]
        nonzero_x = nonzero[1]

        # Sum of y-values in each X's
        y = np.sum(all_combine_binary[int(all_combine_binary.shape[0]/2):, :], axis=0)

        # A peak in the first half, it means the left lane
        # A peak in the second half, it means the right lane
        xhalf = int(y.shape[0]/2)
        xleft_lane = np.argmax(y[:xhalf])
        xright_lane = np.argmax(y[xhalf:]) + xhalf

        # Current positions, index, to be updated for each window - it is argmax x index value in left and right lane
        xleft_current = xleft_lane
        xright_current = xright_lane

        # Create empty lists to receive left and right lane pixel indices, index
        left_lane_idx = []
        right_lane_idx = []

        # Step through the windows one by one
        for window in range(n_windows):
            # Identify window boundaries in x and y (and right and left), those values are all index values
            win_y1 = all_combine_binary.shape[0] - (window+1) * window_height
            win_y2 = all_combine_binary.shape[0] - window * window_height

            win_xleft1 = xleft_current - margin
            win_xleft2 = xleft_current + margin
            win_xright1 = xright_current - margin
            win_xright2 = xright_current + margin
            #print(win_y1, win_y2, win_xleft1, win_xleft2)


            # Identify the nonzero pixels in x and y within the window, nonzero values are index values
            good_left_idx = ((nonzero_y >= win_y1) & 
                              (nonzero_y <= win_y2) & 
                              (nonzero_x >= win_xleft1) &  
                              (nonzero_x <= win_xleft2)).nonzero()[0]
            good_right_idx = ((nonzero_y >= win_y1) & 
                              (nonzero_y <= win_y2) & 
                              (nonzero_x >= win_xright1) &  
                              (nonzero_x < win_xright2)).nonzero()[0]

            # Append these indices, index, to the lists
            left_lane_idx.append(good_left_idx)
            right_lane_idx.append(good_right_idx)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_idx) > minpix:
                xleft_current = np.int(np.mean(nonzero_x[good_left_idx]))
            if len(good_right_idx) > minpix:        
                xright_current = np.int(np.mean(nonzero_x[good_right_idx]))
            #print(left_lane_idx, right_lane_idx)

        # Concatenate the arrays of indices
        left_lane_idx = np.concatenate(left_lane_idx)
        right_lane_idx = np.concatenate(right_lane_idx)

        # Extract left and right line pixel positions
        leftx = nonzero_x[left_lane_idx]
        lefty = nonzero_y[left_lane_idx] 
        rightx = nonzero_x[right_lane_idx]
        righty = nonzero_y[right_lane_idx] 

        # Fit a second order polynomial to each, return values are coefficients
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)
        #print(left_fit, right_fit)
        #print(left_fit.shape, right_fit.shape)

        # Generate x and y values for plotting
        self.ploty = np.linspace(0, all_combine_binary.shape[0]-1, all_combine_binary.shape[0])
        left_fitx = self.left_fit[0] * self.ploty**2 + self.left_fit[1] * self.ploty + self.left_fit[2]
        right_fitx = self.right_fit[0] * self.ploty**2 + self.right_fit[1]  *self.ploty + self.right_fit[2]
        return left_fitx, right_fitx
    
    def fit_next(self, all_combine_binary):
        '''
        This function makes poly fit lines on the left and right if there is previous coefficient values.
        It uses previous coefficient values, left_fit and right_fit, and update there values
        '''
        
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = all_combine_binary.nonzero()
        nonzero_y = nonzero[0]
        nonzero_x = nonzero[1]
        
        margin = 100                   # Set the width of the polynomials, left and right, +/- margin
        
        # Identify region of interest like window using previous poly lane fit, using previous coefficient values
        left_lane_idx = ((nonzero_x >= (self.left_fit[0]*(nonzero_y**2) + 
                                        self.left_fit[1]*nonzero_y + 
                                        self.left_fit[2] - margin)) & 
                         (nonzero_x <= (self.left_fit[0]*(nonzero_y**2) + 
                                        self.left_fit[1]*nonzero_y + 
                                        self.left_fit[2] + margin))) 

        right_lane_idx = ((nonzero_x >= (self.right_fit[0]*(nonzero_y**2) + 
                                         self.right_fit[1]*nonzero_y + 
                                         self.right_fit[2] - margin)) & 
                          (nonzero_x <= (self.right_fit[0]*(nonzero_y**2) + 
                                         self.right_fit[1]*nonzero_y + 
                                         self.right_fit[2] + margin)))  

        # Extract left and right line pixel positions
        leftx = nonzero_x[left_lane_idx]
        lefty = nonzero_y[left_lane_idx] 
        rightx = nonzero_x[right_lane_idx]
        righty = nonzero_y[right_lane_idx]

        # Fits a second order polynomial to each
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)

        # Generates x and y values for plotting
        left_fitx = self.left_fit[0] * self.ploty**2 + self.left_fit[1] * self.ploty + self.left_fit[2]
        right_fitx = self.right_fit[0] * self.ploty**2 + self.right_fit[1] * self.ploty + self.right_fit[2]
        return left_fitx, right_fitx

    def laneFind(self, image):
        '''
        This function is whole pipline for lane lines detection. 
        INPUT: 
            image(image format file, str)
        OUTPUT: 
            return image with Draw Poly Lane Lines in Un-Warped Image
            The image contains Radius of Curvature and Center Offset by meter
        '''
        # Getting bird eyes images (warped) from region of interest on the original image
        # There is M and M_inv: M is matrix, which makes into warped image and 
        #                       M_inv is matrix, which can turn back to original image
        _, warped, M_inv = tran_perspective(image, self.mtx, self.dist)
        img = cv2.cvtColor(warped, cv2.COLOR_BGR2HLS)[:,:,2]
        all_combine_binary = np.zeros_like(img)                          # make empty matrix

        '''
        In order to find the best thresh for each variables (Saturate, Lightness, Gradient(Sobel) 
        Horizontal and Gradient(Sobel) Vertical), I play around with ipywidgets library and get the best thresholds.

        * L-Thresh: 120 
        * S-Thresh: 100 
        * SobelX Min Thresh: 30 
        * Dir. Min Thresh: 10 degree (np.pi/18)
        '''
        # Apply binary Threshold in S-Channel and L-Channel after converted into HLS color space
        _, ls_thresh = ls_channel_thresh(warped, 120, 100)

        # Apply the gradient(sobel) Threshold on the horizontal gradient
        sobel_binary = sobelx_thresh(warped, 30, 255)

        # Apply the gradient direction Threshold to find the direction
        direction_binary = direction_thresh(warped, np.pi/18, np.pi/2)

        combine_thresh = ((sobel_binary == 1) & (direction_binary == 1))
        all_combine_binary[(ls_thresh | combine_thresh)] = 1

        # Draw an image to show the selection window on the out_img, making left and right lane
        out_img = np.dstack((all_combine_binary, all_combine_binary, all_combine_binary))*255
        window_img = np.zeros_like(out_img)

        if self.detected == False:
            left_fitx, right_fitx = Line.fit_initial(self, all_combine_binary)
            self.detected = True
            #print('check 1')
            
        else:
            left_fitx, right_fitx = Line.fit_initial(self, all_combine_binary)
            #print('check 2')

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window = np.array([np.transpose(np.vstack([left_fitx, self.ploty]))])
        right_line_window = np.array([np.flipud(np.transpose(np.vstack([right_fitx, self.ploty])))])
        line_pts = np.hstack((left_line_window, right_line_window))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([line_pts]), [0, 255, 0])

        out_margin = 20
        in_margin = 40
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - out_margin, self.ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + in_margin, self.ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))

        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - in_margin, self.ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + out_margin, self.ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (255, 0, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 0, 255))

        # unwarped from warpperspective image
        y, x = all_combine_binary.shape[:2]
        unwarped = cv2.warpPerspective(window_img, M_inv, (x, y), flags=cv2.INTER_LINEAR)
        image = undistortion(image, self.mtx, self.dist)
        result = cv2.addWeighted(image, 1, unwarped, 1, 0)

        # Calculate curvature in real world, pixel values * pixel per meter
        left_curverad = measure_curvature(all_combine_binary, left_fitx)
        right_curverad = measure_curvature(all_combine_binary, right_fitx)
        lane_mid = (left_fitx[-1] + right_fitx[-1])/2

        # xm_per_pix value, 3.7/700, * different pixels between x-mid(1280/2 = 640) and lane_mid  
        # offset +/-: (+) is offset to the left, (-) is offset to the right
        offset = (3.7/700)*(x/2 - lane_mid)

        text_cur = 'Radius of Curvature: {:04.2f}'.format((left_curverad+right_curverad)/2) + 'm'
        text_offset = 'Center Offset: {:04.2f}'.format(offset) + 'm'
        cv2.putText(result, text_cur , (50, 80), cv2.FONT_HERSHEY_DUPLEX, 1.5, (102, 255, 102), thickness=3)
        cv2.putText(result, text_offset, (50, 140), cv2.FONT_HERSHEY_DUPLEX, 1.5, (102, 255, 102), thickness=3)
        return result