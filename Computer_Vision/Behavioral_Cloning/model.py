import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import math
import os
import re

import cv2

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

def img2vec(X):
    '''
    Convert file name into pixel values
    '''
    for img in range(len(X)):
        for s in range(3):
            X[img][s] = mpimg.imread('./data/'+re.sub(r'[^\S\n\t]+', '', X[img][s]))
    return X

def load_data():
    """
    Load dataset from driving_log and split into training and validation set
    """
    data_df = pd.read_csv('./data/driving_log.csv')

    X = img2vec(data_df[['center', 'left', 'right']].values)
    y = data_df['steering'].values

    Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=.2, random_state=0)

    return Xtr, Xval, ytr, yval

def crop_resize_cvt2yuv(X):
    """
    Crop, resize, covert color in each image to remove sky at the top, and the hood of the car at the bottom
     - Crop: remove top 68, bottom 23 pixels
     - Resize: (160, 320, 3) -> (66, 200, 3), which is inout shape for NVIDIA model
     - Convert Color: RGB to YUV
    """
    for img in range(len(X)):
        for s in range(3):
            X[img][s] = cv2.cvtColor(cv2.resize(X[img][s][68: -23, :, :], (200, 66), cv2.INTER_AREA), cv2.COLOR_RGB2YUV)
    return X

def flip(img, angle):
    """
    The function flipt the image left <-> right, and adjust the steering angle.
    """
    if np.random.randint(-1, 1) < 0:
        #print('check')
        img = cv2.flip(img, 1)
        angle = -angle
    return img, angle

# def get_angle(p0, p1):
#     '''
#     The function calculate the angle based on translate point
#     p0(tuple): original point, (100, 66) == (x, y)
#     p1(tuple): traslated point
#     '''
#     x1, y1 = p0
#     x2, y2 = p1
#     dx = x2 - x1
#     dy = y2 - y1
#     rads = math.radians(90) - math.atan2(-dy, dx)
#     return rads

def translate(img, angle):
    """
    The function makes translating images in random (X, Y) coordinate, between -30.0 and 30.0 for x
                                                                       between -5.0 and 5.0 for y
    And adjust the steering angle 
    """
    #p0 = (100, 66)
    y, x = img.shape[:2]
    tx = np.random.uniform(-30., 30.)
    ty = np.random.uniform(-5., 5.)
    #p1 = (100+tx, 66+ty)
    #angle = get_angle(p0, p1)
    angle += tx * 0.003
    trans_M = np.float32([[1, 0, tx], [0, 1, ty]])
    trans_img = cv2.warpAffine(img, trans_M, (x, y))
    return trans_img, angle

def adjust_gamma(img, angle):
    """
    The function adjust gamma value to change brightness of image
    opencv provides Color map, but changing only brighness with gamma needs to make own color map
    The table is from Google search, mapping the pixel values [0, 255] to adjust gamma values
    """
    gamma = np.random.uniform(0.3, 3.)
    invGamma = 1.0 / gamma
    color_map = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
 
    # apply gamma correction using the color_map
    return cv2.LUT(img, color_map), angle

def shadowing(img, angle):
    """
    The function makes shodow on the images in randomly choosed locations
    Mask: Above the line will be (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
    Since x2 == x1 causes zero-division problem, it needs to make this form (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
    """
    x1, x2 = 200 * np.random.uniform(-1., 1., 2)
    y1, y2 = 0, 66
    xm, ym = np.mgrid[:66, :200]            # make np.array for grid, location of pixles, on the image
    mask = np.zeros_like(img[:, :, 1])
    mask[(ym-y1) * (x2-x1) - (y2-y1) * (xm-x1) > 0] = 1

    # Randomly choose the side, which above/below of the line, has shadow and adjust saturation
    side = mask == np.random.randint(0, 2)
    ratio = np.random.uniform(0.0, 0.3)

    # Adjust Saturation in YUV(Y: brightness of the color)
    img[:,:,0][side] = img[:,:,0][side] * ratio
    return img, angle

def generator(X, y, batch_size, train=1):
    """
    The function generate training images, and make training set with batch_size
    In order to generate images, it needs to various of techniques: flip, translation, a 
    """
    imgSet = np.empty([batch_size, 66, 200, 3])
    angleSet = np.empty(batch_size)
    ### print(imgSet.shape, angleSet.shape)
    while True:
        for idx in np.random.permutation(X.shape[0]):
            n=0
            img, angle = X[idx], y[idx]
            
            '''
            Image Argumentation: 
                - Randomly choose from ['center', 'left', 'right'] images in weight of [0.3, 0.35, 0.35]
                  because distribution of images skewed with straight roads (steering angle == 0.0),
                  so choosing left(steering angle += 0.2) and right(steering angle -= 0.2) in higher rate
                  might help to figure out unblanced input data
                - Making curved images because the images with angle = 0 (straight roads) are enough
                -   
            '''  
            if train:
                l = np.random.choice(['center', 'left', 'right'], p=[0.3, 0.35, 0.35])
                if l == 'center':
                    img, angle = img[0], angle
                elif l == 'left':
                    img, angle = img[1], angle + 0.2         # left side of image has different angle, +0.2 
                else: 
                    img, angle = img[2], angle - 0.2         # right side of image has different angle, -0.2


                if np.random.randint(0, 5) < 4:     # about 80% chance 
                    ran_gen = np.random.choice([flip, translate, shadowing, adjust_gamma])
                    img, angle = ran_gen(img, angle)
            
            else:
                img, angle = img[0], angle
                
            # add the image and steering angle to the batch
            imgSet[n] = img
            angleSet[n] = angle
            
            n+=1
            if n == batch_size:
                break
            
        yield imgSet, angleSet

def bulid_model():
    '''
    Using NVIDIA model
    '''
    model = Sequential()
    
    # Input Normalize - to avoid saturation and make gradients work better
    model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=(66, 200, 3)))
    
    # Add Convolution, Conv2D, layers
    # Apply ELU(Exponential linear unit) function, to take care Vanishing gradient problem
    # Conv2D(output depth, kernel_low, kernel_column, activation_func., subsample=(stride))
    model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(64, 3, 3, activation='elu', subsample=(1, 1)))
    model.add(Conv2D(64, 3, 3, activation='elu', subsample=(1, 1)))
    model.add(Dropout(0.5))         # To avoid overfitting
    
    # Add Fully-Connected layers
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()
    
    return model

def training(model, Xtr, Xval, ytr, yval):
    """
    The function train the NVIDIA model
    """
    checkpoint = ModelCheckpoint('model.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='auto')

    model.compile(loss='mse', optimizer=Adam(lr=1e-4))
    
    model.fit_generator(generator(Xtr, ytr, 40),
                        samples_per_epoch=len(Xtr)*10,
                        nb_epoch=30,
                        max_q_size=1,
                        validation_data=generator(Xval, yval, 40, 0),
                        nb_val_samples=len(Xval),
                        callbacks=[checkpoint],
                        verbose=1)

def main():
    Xtr, Xval, ytr, yval = load_data()
    Xtr = crop_resize_cvt2yuv(Xtr)
    Xval = crop_resize_cvt2yuv(Xval)
    
    model = bulid_model()
    training(model, Xtr, Xval, ytr, yval)

if __name__ == '__main__':
    main()

