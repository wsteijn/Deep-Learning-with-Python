#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  8 13:15:30 2021

@author: willsteijn
"""

#CATS AND DOGS EXAMPLE

#cats and dogs dataset downloaded from www.kaggle.com/c/dogs-vs-cats/data
#copying images to training, validation, and test directories
import os, shutil

#path to directory where the original dataset is uncompressed
original_dataset_dir = 'C:/Users/Will/Downloads/dogs-vs-cats/train/train'
#directory where smaller dataset will be stored
base_dir = 'C:/Users/Will/Downloads/dogs-vs-cats_small'
os.mkdir(base_dir)

#directory for the training data
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
#directory for the validation data
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
#directory for the test data
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

#directory with training cat pictures
train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)

#directory with training dog pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)

#directory with validation cat pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')
os.mkdir(validation_cats_dir)

#directory with validation dog pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)

#directory with test cat pictures
test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)

#directory with test dog pictures
test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)

#copies first 1000 cat images to train_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

#copies the  next 500 images to validation cats directory
fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)

#copies the next 500 images to test_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst) 
    
#copies the frist 1000 dog images to the training dog directory
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)

#copies the next 500 dog images to validation dogs directory
fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)

#copies the next 500 dog images to test dog directory
fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)

#sanity checks
len(os.listdir(train_cats_dir))#1000
len(os.listdir(train_dogs_dir))#1000
len(os.listdir(validation_cats_dir))#500
len(os.listdir(validation_dogs_dir))#500
len(os.listdir(test_cats_dir))#500
len(os.listdir(test_dogs_dir))#500


#building the network
from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation = 'relu' , input_shape = (150, 150, 3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation= 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation  = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))

model.summary()

#configuring the model for training
from keras import optimizers

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr =1e-4), metrics = ['acc'])

#data preprocessing
#steps:
#   1. Read picture files
#   2. decode JPEG content to RGB grids of pixels
#   3. Convert these to floating-point tensors
#   4. Rescale the pixel values (0-255) to the [0,1] interval for neural network processing

from keras.preprocessing.image import ImageDataGenerator

#rescale images by 1/255
train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale= 1./255)

#train dir is target diretory, target size resizes all images to 150x150, class mode is binary because the model uses binary_crossentropy loss
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150,150), batch_size = 20, class_mode = 'binary')
validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(150, 150), batch_size = 20, class_mode='binary')

#view output of generators
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break
#output is batches with 20 samples of 150 x 150 RGB images =(20, 150, 150, 3)
    
#fit the model to the data using the generator with fit_generator
#because the data is being generated endlesly, the model needs to know how many batches to draw from the generator
#before declaring an epoch over - steps_per_epoch argument
#in this case there are 20 samples, so it will take 100 batches until you use all 2000 samples
#validation_steps argment tells the model how many batches to draw from the validation generator for evaluation

history = model.fit_generator(train_generator, steps_per_epoch = 100, epochs = 30, validation_data = validation_generator, validation_steps = 50)
model.save('cas_and_dogs_small_1.h5')

#display loss and accuracy curves during training
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label = 'Training acc')
plt.plot(epochs, val_acc, 'b', label= 'Validation acc')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
#plots are characteristic of overfitting


#Data augmentation
#data augmentation generates more training data from existing training samples by augmenting the samples via a number of random 
#transformations that yield believable-looking images. The goal is that at training time, the model will never see the exact
#same picture twice
#ImageDataGenerator instance
datagen = ImageDataGenerator(rotation_range = 40, width_shift_range = 0.2, height_shift_range = 0.2, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True, fill_mode='nearest')
#rotation range = a range within which to randomly rotate pictures
#width shift and height shift are ranges within which to radonly translate pics vert. or horz.
#shear range is for randomly applying shearing transformations
#zoom range is for randomly zooming inside pics
#horizontal flip is for randomly flipping half the images horizontally - relevant when there are no assumptions of horizontal asymmentry
#fill mode is the strategy used for filling in newly created pixels that may appear after a rotation or a width/height shift

#displaying some randomly augmented training images
from keras.preprocessing import image

fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]

#choose 1 image to augment
img_path = fnames[3]

#read the image and resize it
img = image.load_img(img_path, target_size=(150, 150))

#convert to a nmpy array
x= image.img_to_array(img)

#reshape it to (1, 150,150,3)
x= x.reshape((1,) +x.shape)

#generates batches of randomly transformed images. loops indefinitely so you need to break the loop at some point
i=0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break
plt.show()
#data that is augmented will be highly intercorrelated, so might not be enough to fix overfitting on its own
#for additional help against overfitting, use drop out


#train a model that includes drop out
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])

#training the convnet using data-augmentation generators
train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,)
#validation data shouldnt be augmented
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,target_size=(150, 150),batch_size=32,class_mode='binary')

validation_generator = test_datagen.flow_from_directory(validation_dir,target_size=(150, 150),batch_size=32,class_mode='binary')

history = model.fit_generator(train_generator,steps_per_epoch=100,epochs=30,validation_data=validation_generator,validation_steps=50)
model.save('cats_and_dogs_small_2.h5')


#plotting the training and validation accuracy and validation
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label = 'Training acc')
plt.plot(epochs, val_acc, 'b', label= 'Validation acc')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
