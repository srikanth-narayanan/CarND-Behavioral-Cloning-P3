#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 22:14:36 2017

This python module contains the model implementation for performing behavioural
clonning using the Udacity Unity simulator.

@author: srikanthnarayanan
"""

# importing all necessary libraries
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Convolution2D

def load_dataset(csvpath, imgdir):
    '''
    Function to parse the csv file and corrects its path. The Images are loaded
    using opencv in to image data.
    '''
    csvframe = pd.read_csv(csvpath, names=['center','left','right','steering',
                                           'throttle','brake','speed'])
    csvframe['center'] = csvframe['center'].apply(lambda x: os.path.join(imgdir, os.path.basename(x)))
    imgpath = csvframe['center'].values
    steer = csvframe['steering'].values

    x_train, x_valid, y_train, y_valid = train_test_split(imgpath, steer,
                                                          test_size=0.2,
                                                          random_state=0)
    return x_train, x_valid, y_train, y_valid

def generate_batch(x,y, batch_size=32):
    '''
    Generator to provide sufficient images for training or validation
    '''
    x_new, y_new = shuffle(x, y)
    nsamples = len(x_new)
    
    while 1:
        shuffle(x, y)
        for offset in range(0, nsamples, batch_size):
            batch_x = x_new[offset:offset + batch_size]
            batch_y = y_new[offset:offset + batch_size]
            images=[]
            angles=[]
            for img, ang in zip(batch_x,batch_y):
                new_img = cv2.imread(img)
                new_img = _crop(new_img)
                images.append(new_img)
                angles.append(ang)
            x_set = np.array(images)
            y_set = np.array(angles)
            yield x_set, y_set

def _crop(img):
    '''
    Helper to crop the image
    '''
    top_crop_height = 60
    bottom_crop_height = -25
    return img[top_crop_height:bottom_crop_height,:,:]

def train_model(x_train, x_valid, y_train, y_valid):
    '''
    Keras model to train the images.
    '''
    train_gen = generate_batch(x_train, y_train, batch_size=32)
    validation_gen = generate_batch(x_valid, y_valid, batch_size=32)

    row, col, ch = 75, 320, 3
    model = Sequential()
    model.add(Lambda(lambda x:x/127.5 -1., input_shape=(row,col,ch)))
    model.add(Convolution2D(6,5,5,activation="relu"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6,5,5,activation="relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    
    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(train_gen, samples_per_epoch=len(x_train), 
                        validation_data=validation_gen,
                        nb_val_samples=len(x_valid), nb_epoch=5,verbose=1)
    model.save('model.h5')
    
def main():
    '''
    main function to run the training process.
    '''
    
    csvpath = r"/home/carnd/bhclone/data/driving_log.csv"
    imgdir = r"/home/carnd/bhclone/data/IMG"
    x_train, x_valid, y_train, y_valid = load_dataset(csvpath, imgdir)
    train_model(x_train, x_valid, y_train, y_valid)
    
    import gc
    gc.collect()
    
    print("Garbage Collected.... Keras Session Cleared!")
if __name__ == "__main__":
    main()
