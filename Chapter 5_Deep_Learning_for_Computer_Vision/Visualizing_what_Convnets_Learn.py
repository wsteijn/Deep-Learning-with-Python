#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  8 13:26:08 2021

@author: willsteijn
"""

#visualizing intermediate activations
from keras.models import load_model
#load model created in Cats_and_Dogs_Pretrained_Convnet file
model = load_model('cats_and_dogs_small_2.h5')
model.summary()

#preprocessing a single image
img_path = '/Users/Will/Downloads/dogs-vs-cats_small/test/cats/cat.1700.jpg'
#preprocess the image into a 4D tensor
from keras.preprocessing import image
import numpy as np
img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255. #remember that the model was trained on inputs that were preprocessed this way
# Its shape is (1, 150, 150, 3)
print(img_tensor.shape)

#display the test picture
import matplotlib.pyplot as plt

plt.imshow(img_tensor[0])
plt.show

#in order to extract the feature maps, create a Keras model that takes batches of images as inputs, and outputs the activactions of all conv and pooling layers
#instantiating a model from an input tensor and a list of input tensors
from keras import models 
#extract the outputs of the top 8 layers
layer_outputs = [layer.output for layer in model.layers[:8]]
#create a model that will return layer_outputs given the model input
activation_model = models.Model(inputs = model.input, outputs = layer_outputs)
#return a list of five numpy arrays: one array per layer activation
activations = activation_model.predict(img_tensor)

#activation of the first convolution layer for the cat image input
first_layer_activation = activations[0]
first_layer_activation.shape
#(1, 148, 148, 32): its a 148 x 148 feature map with 32 channels
#plot the fourth channel of the activation of the first layer of the original model
import matplotlib.pyplot as plt
plt.matshow(first_layer_activation[0, :, :, 4], cmap = 'viridis')
#plot the 7th channel
plt.matshow(first_layer_activation[0, :, :, 7], cmap = 'viridis')

#visualizing ever channel in every intermediate activation
#names of the layers, so you can have them as part as your plot
layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

images_per_row = 16

#display the feature maps
for layer_name, layer_activation in zip(layer_names, activations):
    #number of features in the feature map
    n_features = layer_activation.shape[-1]
    #the feature map has shape (1, size, size, n_features)
    size = layer_activation.shape[1]
    #tile the activation channels in the matrix
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    #til each filter into a bit horizontal grid
    for col in range(n_cols):
        for row in range(images_per_row):
            #post-process the feature to make it visually palatable
            channel_image = layer_activation[0,:, :,col * images_per_row + row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            #display the grid
            display_grid[col * size : (col + 1) * size,row * size : (row + 1) * size] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
#first layer acts as a collection of various edge detectors
#as you go higher, the activations become increasingly abstract
#sparsity of the activations increases with the depth of the layer

#visualizing convnet filters
#display the visual pattern that each filter is meant to respond to
#done with gradient ascent in input space - to maximize the response of a specific filter, starting from a blank input image
#the resulting input image will be one that the chosen filter is maximally responsive to
    
#defining the loss tensor for filter visualization
from keras.applications import VGG16
from keras import backend as K 
model = VGG16(weights='imagenet',include_top=False)
layer_name = 'block3_conv1'
filter_index = 0
layer_output = model.get_layer(layer_name).output
loss = K.mean(layer_output[:, :, :, filter_index])

#obtaining the gradient of the loss with regard to the input
#call to gradients to return a list of tensors (of size 1 in this case). Keep only the first element, which is the tensor ([0])
grads = K.gradients(loss, model.input)[0] 

#normalize gradient by dividing by the L2 norm
grads = (K.sqrt(K.mean(K.square(grads)))+ 1e-5) #add 1e-5 before dividing to avoid accidentally dividing by 0

#fetching Numpy output values given Numby input values
iterate = K.function([model.input], [loss, grads])

import numpy as np
loss_value, grads_value = iterate([np.zeros((1, 150, 150, 3))])

#loss maximization via stochastic gradient descent
#start from a gray image with some noise
input_img_data = np.random.random((1, 150, 150, 3)) * 20 + 128.
#magnitude of each gradient update
step = 1
#run gradient ascent for 40 steps
for i  in range(40):
    #compute the loss value and gradient value
    loss_value, grads_value = iterate([input_img_data])
    #adjust the input image in the direction that maximizes the loss
    input_img_data += grads_value * step
#the resulting image tensor is a floating point tensor of shape (1, 150, 150, 3) with values that  may not be integers w/in (0, 255)

#utility funciton to converat a tensor into a valid image
def deprocess_image(x):
    #normalize the tensor: center on 0, ensure that std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    #clip to [0,1]
    x += 0.5
    x = np.clip(x, 0, 1)
    #converts to an RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

#function to gernate filter visualizations
def generate_pattern(layer_name, filter_index, size=150):
    #build a loss function that maximize the activation of the nth filter of the layer under consideration
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])
    #compute the gradient of the input picutre w/ regard to this loss
    grads = K.gradients(loss, model.input)[0]
    #normalization trick: normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    #returns the loss and grads given the input picture
    iterate = K.function([model.input], [loss, grads])
    #start from a gray image with some noise
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.
    #run gradient ascent for 40 steps    
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
    img = input_img_data[0]
    return deprocess_image(img)

plt.imshow(generate_pattern('block3_conv1',0))
#filter 0 in layer block3_conv1 is responsive to a polka-dot pattern

#can visualize every filter in every layer
#start by looking at the first 64 filters in each layer, and look at the first layer of each convolution block
size = 64
margin = 5
#emptly black image to store results
results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))
#iterate over the rows of the results grid
for i in range(8):
    #iterates over the columns of the results grid
    for j in range(8):
        #generates the pattern for filter i + (j *8) in layer_name
        filter_img = generate_pattern(layer_name, i + (j * 8), size=size)
        #puts the result in the square (i, j) of the results grid
        horizontal_start = i * size + i * margin
        horizontal_end = horizontal_start + size
        vertical_start = j * size + j * margin
        vertical_end = vertical_start + size
        results[horizontal_start: horizontal_end,vertical_start: vertical_end, :] = filter_img
plt.figure(figsize=(20, 20))
plt.imshow(results)
#each layer in a convnet learns a collection of filters such that their inputs can be expressed as combinations of the filters
#filter from first layer in the model (block1_conv1) ecode siple directional edges and colors
#filters from block2_conv1 encode simple textures made from combinations of edges and colors
