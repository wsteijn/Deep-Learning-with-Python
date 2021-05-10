#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 20:55:57 2021

@author: willsteijn
"""


#SEQUENCE PROCESSING WITH COVNETS
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import layers
from keras.models import Sequential
max_features = 10000
max_len = 500
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), len(x_test))
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

#training and evaluating a simple 1D convnet on the IMDB data
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
model = Sequential()
model.add(layers.Embedding(max_features, 128, input_length=max_len))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))
model.summary()
model.compile(optimizer=RMSprop(lr=1e-4),loss='binary_crossentropy',metrics=['acc'])
history = model.fit(x_train, y_train,epochs=10,batch_size=128,validation_split=0.2)


#because 1D convnets process input batches independently, they are not sensitive to the order of timesteps beyond a local scale(size of the convolution windows)
#to recorgnize longer-term patterns you can stack many convolution layers and pooling layers, resultingin upper ayers that will see long chunks of the original data
#however, this doenst work very well 

#on temperature dataset - because more recent data points should be interpreted differently from older data points, the convnet fails at producing meaningful results

#one strategy to combine the speed and lightness of convnets with the order-sensitivity of RNNs is to use a 1D convnet as a preprocessing step before an RNN
#
#preparing higher-resolution data generators for the Jena dataset
step = 3
lookback = 720
delay = 144

train_gen = generator(float_data,lookback=lookback,delay=delay,min_index=0,max_index=200000,shuffle=True,step=step)
val_gen = generator(float_data,lookback=lookback,delay=delay,min_index=200001,max_index=300000,step=step)
test_gen = generator(float_data,lookback=lookback,delay=delay,min_index=300001,max_index=None,step=step)
val_steps = (300000 - 200001 - lookback) // 128
test_steps = (len(float_data) - 300001 - lookback) // 128

from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
model = Sequential()
model.add(layers.Conv1D(128, 5, activation='relu',input_shape=(None, float_data.shape[-1])))
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(128, 5, activation='relu'))
model.add(layers.GRU(128, dropout=0.1, recurrent_dropout=0.2))
model.add(layers.Dense(1))
model.summary()
model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,steps_per_epoch=500,epochs=20,validation_data=val_gen,validation_steps=val_steps)
#if getting loss: nan, increase batch size (now at 128) and/or decrease dropout rate
#results arent as good as regularized GRU alone, but is significantly faster
