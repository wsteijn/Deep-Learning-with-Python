#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  8 13:30:52 2021

@author: willsteijn
"""

#Visualizing heatmaps of class activation
from keras import backend as K 
import matplotlib.pyplot as plt

#Class Activation Map visualization - heatmaps of class activation over input images
#indicates how important each location is with respect to the class under consideration
#demonstrate technique using pretrained VGG16 network
from keras.applications.vgg16 import VGG16
model = VGG16(weights = 'imagenet')

#preprocessing an inputimage for vgg16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
#download image of elephants from figure 5.34 on page 173
#local path to target image
img_path = '/Users/Will/Downloads/creative_commons_elephant.jpg'
#pthon image libarary image of size 224 x 224
img = image.load_img(img_path, target_size=(224, 224))
#float32 numpy array of shape (224, 224, 3)
x = image.img_to_array(img)
#add dimenion to transform the array into a batch size of (1, 224, 224, 3)
x = np.expand_dims(x, axis=0)
#preprocess the batch
x = preprocess_input(x)

#now run the pretrained network on the image and decode its prediction vector 
preds = model.predict(x)
print('Predicted:' , decode_predictions(preds, top = 3)[0])

np.argmax(preds[0])
#386 - african elephant class is at index 386

#set up the Grad-Cam algorithm
#african elephant entry in the prediction vector
african_elephant_output = model.output[:, 386]
#output feature map of the block5_conv3 layer, the last convolution layer in VGG16
last_conv_layer = model.get_layer('block5_conv3')
#gradient of the African elephant class with regar to the ouptu feature map of block5_conv3
grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]
#vector of shape (512,), where each entry is the mean intensity of the gradient over a specific feature-map channel
pooled_grads = K.mean(grads, axis=(0, 1, 2))
#access the values of the quantities you just defined: pooled_grads and the output feature map of block5_conv3, given a sample image
iterate = K.function([model.input],[pooled_grads, last_conv_layer.output[0]])
#values of two quantities, as numpy arrays, given the sample image of two elephants
pooled_grads_value, conv_layer_output_value = iterate([x])
#mulitpy each channel in the feature-map array by 'how important this channel is' with regard to the 'elephant' class
for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
#channel-wise mean of the resulting feature map is the heatmap of the class activation
heatmap = np.mean(conv_layer_output_value, axis=-1)

#heatmap post-processing
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)

#superimpose the heatmap with the original picture
import cv2
#use cv2 to load the original image
img = cv2.imread(img_path)
#resize the heatmap to be the same size as the original image
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
#convert heatmap to rgb
heatmap = np.uint8(255 * heatmap)
#apply heatmap to original image
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#0.4 here is a heatmap intensity factor
superimposed_img = heatmap *0.4 +img
#saveimage to disk
cv2.imwrite('/Users/Will/Downloads/elephant_cam.jpg', superimposed_img)