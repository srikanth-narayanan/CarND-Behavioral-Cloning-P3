#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 10:23:52 2017

@author: srikanthnarayanan

An Image processing utility for Conv Net
"""
import cv2
import numpy as np
from sklearn.utils import shuffle


def readimage(image):
    '''
    function to read image from file and send it as RGB matrix
    '''
    img = cv2.imread(image)
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img_RGB


def cropimage(image):
    '''
    function to crop a given image for given region
    '''
    top_crop_height = 60
    bottom_crop_height = -25

    return image[top_crop_height:bottom_crop_height, :, :]


def resizeimage(image):
    '''
    function to resize the image the size needed by the nvidia conv model
    '''
    image_width = 200
    image_height = 66
    # The number of channel stays as 3

    return cv2.resize(image, (image_width, image_height),
                      interpolation=cv2.INTER_AREA)


def changecolor(image, color="YUV"):
    '''
    function to change color space. Nvidia recommends a YUV color Space
    '''
    if color == "YUV":
        img = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    elif color == "HSV":
        img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    elif color == "LUV":
        img = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
    elif color == "BGR":
        img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        print("Unknown Color Space")
        img = image

    return img


def brightness_change(image):
    '''
    function to change brightness of the image
    '''
    # Convertion to HSV, where v is value and also referred to as brightness
    # https://en.wikipedia.org/wiki/HSL_and_HSV
    # picking a random brightness ration
    br_ratio = np.random.uniform(0.1, 1.2)
    hsb = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # change the three axis of the 3rd dimension
    hsb[:, :, 2] = hsb[:, :, 2] * br_ratio
    img = cv2.cvtColor(hsb, cv2.COLOR_HSV2RGB)

    return img


def flip_image(image, steer_angle):
    '''
    function to flip image
    '''
    img = cv2.flip(image, 1)  # Vertical flip
    steer_angle = -steer_angle

    return img, steer_angle


def translate_image(image, steer_angle, xlimit, ylimit):
    '''
    translate image within a random value
    '''
    xlimit = np.random.uniform(-xlimit, xlimit)
    ylimit = np.random.uniform(-ylimit, ylimit)
    nrow, ncol = np.shape(image)[:2]
    trans_matrix = np.float32([[1, 0, xlimit], [0, 1, ylimit]])
    trans_img = cv2.warpAffine(image, trans_matrix, (ncol, nrow))
    steer_angle += xlimit * 0.002

    return trans_img, steer_angle


def batch_gen(x, y, batch_size=32):
    '''
    Generator to provide sufficient images for training or validation
    '''
    x_new, y_new = shuffle(x, y)
    nsamples = len(x_new)

    while True:
        for offset in range(0, nsamples, batch_size):
            batch_x = x_new[offset:offset + batch_size]
            batch_y = y_new[offset:offset + batch_size]
            images = []
            angles = []
            for img, ang in zip(batch_x, batch_y):
                # preprocess image
                new_img = readimage(img)
                new_img = cropimage(new_img)
                new_img = resizeimage(new_img)
                new_img = changecolor(new_img)
                # augument image
                new_img = brightness_change(new_img)
                new_img, new_ang = flip_image(new_img, ang)
                new_img, new_ang = translate_image(new_img, new_ang, 90, 9)
                images.append(new_img)
                angles.append(new_ang)
            x_set = np.array(images)
            y_set = np.array(angles)
            yield x_set, y_set
