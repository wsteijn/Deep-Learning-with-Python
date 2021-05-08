#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  8 12:54:58 2021

@author: willsteijn
"""

#small covnet on MNIST data

#start with mnist data and normal dense layers
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#preparing the image  data
train_images = train_images.reshape((60000, 28*28))
train_images=train_images.astype('float32') /255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

#preparing the labels
from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#building the model
from keras import models
from keras import layers
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape = (28*28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

#fit the model
network.fit(train_images, train_labels, epochs=5, batch_size=128)
test_loss, test_acc = network.evaluate(test_images, test_labels)
test_acc # .979999

#now try with a convnet
from keras import layers
from keras import models