#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 20:54:45 2021

@author: willsteijn
"""

#BIDIRECTIONAL RNNs
#processes a sequence in both ways - can catch patterns that may be overlooked by a unidirectional RNN

#training and evaluating an LSTM using reversed seqences
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import layers
from keras.models import Sequential
max_features = 10000
maxlen = 500
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = [x[::-1] for x in x_train]
x_test = [x[::-1] for x in x_test]
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
model = Sequential()
model.add(layers.Embedding(max_features, 128))
model.add(layers.LSTM(32))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
history = model.fit(x_train, y_train,epochs=10,batch_size=128,validation_split=0.2)
#get very similar performance to that of the chronological-order LSTM
#on text datasets, reversed-order processing generally works as well as chronological processing

#training and evaluating a bidirectional LSTM
model = Sequential()
model.add(layers.Embedding(max_features, 32))
model.add(layers.Bidirectional(layers.LSTM(32)))
model.add(layers.Dense(1, activation = 'sigmoid'))
model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['acc'])
history = model.fit(x_train, y_train, epochs = 10, batch_size = 128, validation_split = 0.2)
#slightly better than regular LSTM, overfits more quickly 
#with some regularization, the bidrectional approach would likely be a strong performer

#training a bidirectional GRU on the temperature data
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
model = Sequential()
model.add(layers.Bidirectional(layers.GRU(32), input_shape=(None, float_data.shape[-1])))
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,steps_per_epoch=500,epochs=40,validation_data=val_gen,validation_steps=val_steps)
#performs about as well as regular GRU layer - makes sense b/c all predictive capacity comes from the chronological half of the network
#further improvements could come from:
#   adjusting number of units in each recurrent layer 
#   adjust the learning rate used by the RMSprop optimzer
#   try using LSTM layers instead of GRU
#   try using a bigger densely connected regressor on top of the recurrent layers: meaning a bigger Dense layer or a stack of Dense layer
#   dont forget to run best-performing models on the test set

#future study - recurrent attention and sequence masking

