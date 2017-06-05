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
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from imagemanager import batch_gen
import argparse


def load_dataset(csvpath, imgdir):
    '''
    Function to parse the csv file and corrects its path. The Images are loaded
    using opencv in to image data.
    '''
    steering_offset = 0.1
    csvframe = pd.read_csv(csvpath, names=['center', 'left', 'right',
                                           'steering', 'throttle', 'brake',
                                           'speed'])

    def _pathjoin(fpath):
        '''
        helper for os path modification
        '''
        return os.path.join(imgdir, os.path.basename(fpath))

    csvframe['center'] = csvframe['center'].apply(_pathjoin)
    csvframe['left'] = csvframe['left'].apply(_pathjoin)
    csvframe['right'] = csvframe['right'].apply(_pathjoin)

    img_center = csvframe['center'].values
    steer_center = csvframe['steering'].values

    img_left = csvframe['left'].values
    steer_left = csvframe['steering'].values + steering_offset

    img_right = csvframe['right'].values
    steer_right = csvframe['steering'].values - steering_offset

    # Concatenate all
    image_all = np.concatenate((img_center, img_left, img_right), axis=0)
    steer_all = np.concatenate((steer_center, steer_left, steer_right), axis=0)

    x_train, x_valid, y_train, y_valid = train_test_split(image_all, steer_all,
                                                          test_size=0.2,
                                                          random_state=0)
    return x_train, x_valid, y_train, y_valid


def train_model(x_train, x_valid, y_train, y_valid):
    '''
    This function implements the NVIDIA model as described in the paper
    end to end deep learning for self driving cars.
    https://arxiv.org/pdf/1604.07316v1.pdf
    '''
    BATCH = 64
    train_gen = batch_gen(x_train, y_train, batch_size=BATCH)
    validation_gen = batch_gen(x_valid, y_valid, batch_size=BATCH)

    HEIGHT, WIDTH, CHANNELS = 66, 200, 3
    SHAPE = (HEIGHT, WIDTH, CHANNELS)
    N_SAMPLES = len(x_train)
    LEARN_RATE = 1e-4

    model = Sequential()

    # Normalisation layer to the input shape
    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=SHAPE))

    # 1st Convolution layer with output of 24. kernel (5,5)
    model.add(Conv2D(24, 5, 5, activation="elu", strides=(2, 2)))

    # 2nd Convolution layer with output of 36. kernel (5,5)
    model.add(Conv2D(36, 5, 5, activation="elu", strides=(2, 2)))

    # 3rd Convolution layer with output of 48. kernel (5,5)
    model.add(Conv2D(48, 5, 5, activation="elu", strides=(2, 2)))

    # 4th Convolution layer with output of 64. kernel (3,3)
    model.add(Conv2D(48, 3, 3, activation="elu"))

    # 5th Convolution layer with output of 64. kernel (3,3)
    model.add(Conv2D(48, 3, 3, activation="elu"))

    # 6th Flatten
    model.add(Flatten())

    # 7th Dense or Fully Connected
    model.add(Dense(100, activation="relu"))

    # 8th Dense or Fully Connected
    model.add(Dense(50, activation="relu"))

    # 9th Dense or Fully Connected
    model.add(Dense(10, activation="relu"))

    # compilation of the model with ADAM optimiser
    model.compile(loss='mse', optimizer=Adam(lr=LEARN_RATE))

    # Run model using the generators for images
    model.fit_generator(train_gen, samples_per_epoch=N_SAMPLES,
                        validation_data=validation_gen,
                        nb_val_samples=len(x_valid), nb_epoch=5, verbose=1)

    # Save model
    model.save('model.h5')


def main():
    '''
    main function to run the training process.
    '''
    parser = argparse.ArgumentParser(description="Behavioural Clonning")
    parser.add_argument('-f', help='csv file path', dest='csvpath', type='str')
    parser.add_argument('-d', help='img directory', dest='imgdir', type='str')

    # parse arguments
    args = parser.parse_args()
    csvpath = args.csvpath
    imgdir = args.imgdir

    # load dataset and train model
    x_train, x_valid, y_train, y_valid = load_dataset(csvpath, imgdir)
    train_model(x_train, x_valid, y_train, y_valid)

    # Explicit garbage collection
    import gc
    gc.collect()
    print("Garbage Collected.... Keras Session Cleared!")

if __name__ == "__main__":
    main()
