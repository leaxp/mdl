# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense
import os
import numpy as np
from skimage import io

class LeNet:

    def __init__(self, input_shape, conv_1, pool_1, conv_2, pool_2, hidden,
                 classes):
        self.model = Sequential()
        # first set of CONV => RELU => POOL
        self.model.add(Conv2D(*conv_1, padding='same', activation='relu',
                              data_format='channels_last',
                              input_shape=input_shape))
        self.model.add(MaxPooling2D(pool_1[0], pool_1[1]))
        # second set of CONV => RELU => POOL
        self.model.add(Conv2D(*conv_2, padding='same', activation='relu',
                              data_format='channels_last'))
        self.model.add(MaxPooling2D(pool_2[0], pool_2[1]))
        # set of FC => RELU layers
        self.model.add(Flatten())
        self.model.add(Dense(hidden, activation='relu'))
        # softmax classifier
        self.model.add(Dense(classes, activation='softmax'))

def load_images(folder):
    images = []
    labels = []
    for file in os.listdir(folder):
        if file.endswith(".png"):
            images.append(io.imread(folder + file, as_grey=True))
            if file.find("einstein") > -1:
                labels.append(1)
            elif file.find("curie") > -1:
                labels.append(2)
            elif os.path.splitext(file)[0].isdigit():
                labels.append(int(os.path.splitext(file)[0]))
            else:
                labels.append(0)
    return images, labels

def deshear(filename):
    image = io.imread(filename)
    distortion = image.shape[1] - image.shape[0]
    shear = tf.AffineTransform(shear=math.atan(distortion/image.shape[0]))
    return tf.warp(image, shear)[:, distortion:]


def normalize_images(images):
    for i in range(len(images)):
        images[i] = images[i][0:100, 0:100]
        images[i] = images[i]/np.amax(images[i])
    return np.array(images)
