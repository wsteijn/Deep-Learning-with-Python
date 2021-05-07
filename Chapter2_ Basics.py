#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 17:27:11 2021

@author: willsteijn
"""
#conda install keras

from keras.datasets import mnist

#train images and train labels form the training set - the data the model will learn from
#test images and test labels form the test set - the data the model will be tested on
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#explore training data
train_images.shape #(60000,28,28)
len(train_labels) # 60000
train_labels

#explore test data
test_images.shape #(10000,28,28)
len(test_labels) #10000
test_labels

#The network architecture
from keras import models
from keras import layers

#Build the model
#layer = data-processing module that you can think of as a filter for the data
#   some data goes in, and it comes out in a more useful form
#   most of deep learning consists of chaining together simple layers that will implement a form of progessive data distillation
network = models.Sequential()
#Dense = fully connected
#each layer applies a few siple tensor operations to the input data, operations involve weight tensors
network.add(layers.Dense(512, activation= 'relu', input_shape = (28 * 28,)))
#10-way softmax layer, which means it will return an array of 10 probability scores (summing to 1)
network.add(layers.Dense(10, activation='softmax'))


#Compilation step
#to make network ready for training, need three more things:
#1. loss function = how the network will be able to measure its performance on the training data, used as a feedback signal for learning the weight tensors and which training will attempt to minimize
#2. optimizer = the mechanism through which the network will update itself based on the data it sees and the loss function, define exact rules for specific use of gradient descent
#3. metrics to monitor during training and testing = here, we care only about accuracy (fraction of images that were correctly classified)
network.compile(optimizer= 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Prepare the data
#before training, preprocess the data by reshaping it into the shape the network expects for scaling so that
#all values are in the [0, 1] interval. Transform our training images from an array of shape (6000, 28, 28)
#of type uint8 with values in [0, 255] interval to a float32 array of shape (6000, 28 * 28) w/ values between 0 and 1.
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32')/255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32')/255

#categorically encode the labels
from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#Fit the model to its training data
#network will iterate on the training data in mini-batches of 128 samples, 5 times over
#at each iteration, the network will compute the gradients of the weights with regard to the loss on the batch, and update the weights accordingly
network.fit(train_images, train_labels, epochs = 5, batch_size = 128)

#check model statistics on the test data
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)


#Data representations
import numpy as np

#scalars (0D tensors)
x = np.array(12)
x #array(12)
# a scalar tensor has 0 axes
x.ndim #0

#vectors (1D tensors)
x = np.array([12, 3, 6, 14])
x #array ([12, 3, 6, 14])
x.dim #1

#matrices (2D tensors)
x = np.array([[5, 78, 2, 34, 0],
                  [6, 79, 3, 35, 1],
                  [7, 80, 4, 36, 2]])
x.ndim #2

#3D tensors and higher dimensional tensors

x = np.array([[[5, 78, 2, 34, 0],
                   [6, 79, 3, 35, 1],
                   [7, 80, 4, 36, 2]],
                [[5, 78, 2, 34, 0],
                   [6, 79, 3, 35, 1],
                   [7, 80, 4, 36, 2]],
                [[5, 78, 2, 34, 0],
                   [6, 79, 3, 35, 1],
                   [7, 80, 4, 36, 2]]])
x.ndim #3

from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#display the number of axes of the tensor train_images
print(train_images.ndim) #3
#print the shape
print(train_images.shape)
#print the data type
print(train_images.dtype)
#we have a 3D tensor of 8-bit integers 


#Display the 4th digit
digit = train_images[4]

import matplotlib.pyplot as plt
plt.imshow(digit, cmap = plt.cm.binary)
plt.show()


#Manipulating tenors in Numpy

#select digits #10 to #100 and put in an array
my_slice = train_images[10:100]
print(my_slice.shape) #(90, 28, 28)

#equivalent to this more detailed notation which specifies a start index and stop index for each tensor acis
my_slice = train_images[10:100, :, :] #(90, 28, 28)
my_slice = train_images[10:100, 0:28, 0:28] #(90, 28, 28)

#select 14 x 14 pixels in the bottom right corner of all images
my_slice = train_images[:, 14:, 14:]

#negative indices are also possble. crop the images to patches of 14 x 14 pixels centered in the middle
my_slice = train_images[:, 7:-7, 7:-7]


#The notion of batches

#in general, the first axis (axis 0) in all data tensors will be the samples axis
#deep learning models dont process an entire dataset at once, it is broken into batches
batch_1 = train_images[:128]
batch_2 = train_images[128:256]
#nth batch
batch_n = train_images[128*n:128*(n+1)]


#The gears of neural networks: tensor operations

#all transformations learned by deep NN can be reduced to a handful of tensor operations applied to tensors of numeric data
keras.layers.Dense(512, activation = 'relu')
#this layer is interpreted as a function, which takes as input a 2D tensor, and returns another 2D tensor as a new representation for the tensor input
#output = relu(dot(W, input) + b)
#relu(x) = max(x,0)
#in numpy, you can do element-wise operation
z = x + y #element-wise addition
z = np.maximum(z, 0.) #element-wise relu


#Broadcasting

#what happens with addition when the shapes of two tensors being added differ?
#smaller tensor will be broadasted to match the shape of the larger tensor
#1. axes are added to the smaller tensor to match the ndim of the larger tensor
#2. smaller tensor is repeated alonside these new axes to match the full shape of the larger tensor
#naive implementation
#x is a 2D numpy tensor
def naive_add_matrix_and_vector(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    #y is a numpy vector
    assert x.shape[1] == y.shape[0]
    #avoid overwriting the input tensor
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[j]
    return x

#the following applies element-wise maximum operation to two densors of different shapes via broadcasting
x = np.random.random((64, 3, 32, 10))
y = np.random.random((32, 10))
#output has shape (64, 3, 32, 10), like x
z = np.maximum(x, y)
print(z.shape)


#Tensor dot

#also called tensor product, combines entries in the input tensors
z = np.dot(x,y)
#the dot product between two vectors is a scalar, and only vectors with the same number of elements are
#compatible for a dot product
def naive_vector_dot(x, y):
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0]
    z = 0.
    for i in range(x.shape[0]):
        z += x[i] * y[i]
    return z
#dot product between maxtrix x and vector y - returns a vector where the coefficients are the dot products
#between y and the rows of x
#dot product between two matrices - iff x.shape[1] == y.shape[0]
#result is a matrix with shape (x.shape[0], y.shape[1])
#coefficients are vector products between rows of x and columns of y
#naive implementation
def naive_matrix_dot(x, y):
    #x and y are Numpy matrices
    assert len(x.shape) == 2
    assert len(y.shape) == 2
    #first dimension of x must be the same as the 0th dimension of y
    assert x.shape[1] == y.shape[0]
    #create matrix of 0s w/ specific shape
    z = np.zeros((x.shape[0], y.shape[1]))
    #iterate over rows of x
    for i in range(x.shape[0]):
        #iterate over columns of y
        for j in range(y.shape[1]):
            row_x = x[i, :]
            column_y = y[:, j]
            z[i, j] = naive_vector_dot(row_x, column_y)
    return z


#Tensor reshaping

#reshaping a tensor means rearranging its rows and columns to match a target shape
#reshaped tensor has same number of total coefficients as the initial tensor
x = np.array([[0., 1.],
              [2., 3.],
              [4., 5.]])
print(x.shape) #(3,2)

x = x.reshape((6,1))
print(x.shape) #(6,1)

x = x.reshape((2,3))
print(x.shape) #(2,3)

#transposition = exchaning its rows and its columns so that x[i,:] becomes x[:, i]
x = np.zeros((300,20))
x = np.transpose(x)
print(x.shape) #(20, 300)


#Engine of neural networks: gradient-based optimization

output = relu(dot(W, input) + b)
#W and b are tensors that are attributes of the layer - theyre weights or trainable paramaters of the layer
#(kernel and bias attributes)
#initially, weight matrices are filled w/ smal random values - gradually adjusted based on a feedback signal
#steps in training loop:
#1. draw a batch of training samples x and corresponding targets y
#2. run network on x to obtain predictions
#3. compute loss of network on the batch (meaure of mismatch between y and predictions)
#4. update all weights of network in a way that slightly reduces the loss on this batch
#for #4 - take advantage of all operations in network are differentiable and compute the gradient of loss w/ regard to the network's coeff.
#5. move the paramaters a little in the opposite direction from the gradient to reduce loss on the batch
#avoid local minima using momentum = move each step based not only on current slope, but also current velocity (resulting from past slope)
#naive implementation
past_velocity = 0. 
momentum = 0.1
while loss > 0.01:
    w, loss, gradient = get_current_parameters()
    velocity = past_velocity * momentum + learning_rate * gradient
    w = w + momentum * velocity - learning_rate * gradient
    past_velocity = velocity
    update_parameter(w)

#Backpropagation

#in practice, a neural network function consists of many tensor operations chained together, each w/ a simple, known derivative
#apply the chain rule to the computaiton of the gradient values off a neural network = backpropagation
#use symbolic differentiation = compute a gradient function for the chain that maps network parameter values to gradient values    
    
    
