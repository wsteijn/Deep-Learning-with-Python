#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 20:52:54 2021

@author: willsteijn
"""

#RECURRENT NEURAL NETWORKS

#psuedocode RNN
#state_t = 0 #the state at t
#for input_t in input_sequence: #iterates over sequence elements
    #output_t = activation(dot(W,input_t) + dot(U, state_t) + b
    #state_t = output_t #the previous ouput becomes the state for the next iteration

    
#Numpy implementation of a simple RNN
import numpy as np

timesteps = 100 #number of timesteps in the input sequence
input_features = 32 #dimensionality of the input feature space
output_features = 64 #dimensionality of the output feature space

#input data: random noise for use in this example
inputs = np.random.random((timesteps, input_features))

#initial state: an all-zero vector
state_t = np.zeros((output_features,))

#create  random weight matrices
W = np.random.random((output_features, input_features))
U = np.random.random((output_features, input_features))
b = np.random.random((output_features))

successive_outputs = []
#input_t is a vector of shape (input_features,)
for input_t in inputs:
    #combine the input with the current state (the previous output) to obtain the current output
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
    #store output in a list
    successive_outputs.append(output_t)
    #update the state of the network for the next timestep
    state_t = output_t
#the final output is a 2D tensor of shape (timesteps, output_features)
final_output_sequence = np.concatenate(successive_outputs, axis = 0)


#recurrent layer in Keras
from keras.layers import SimpleRNN

#inputs of shape (batch_size, timesteps, input_features)
#can return the full sequences of successive outputs for each timestep (batch_size, timesteps, output_features)
#or can return onlly the last oupput for each input sequence (batch_size, output_features)

#returns only the output at the last timestep
from keras.models import Sequential
from keras.layers import Embedding

model = Sequential()
model.add(Embedding(1000, 32))
model.add(SimpleRNN(32))
model.summary()

#return the full state sequence
model = Sequential()
model.add(Embedding(1000, 32))
model.add(SimpleRNN(32, return_sequences = True))
model.summary()

#can be useful to stack several recurrent layers one after the other in order to increase the representational power of each network
#have to get all of the immediate layers to return full sequence outputs
model = Sequential()
model.add(Embedding(1000, 32))
model.add(SimpleRNN(32, return_sequences = True))
model.add(SimpleRNN(32, return_sequences = True))
model.add(SimpleRNN(32, return_sequences = True))
model.add(SimpleRNN(32)) #last layer only returns the last output
model.summary()

#use RNN on IMDB movie-review classification

#prepare the IMDB data
from keras.datasets import imdb
from keras.preprocessing import sequence

max_features = 10000 # number of words to consider as features
maxlen= 500 #cut off text after this many max_feature words
batch_size = 32

(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words = max_features)

print(len(input_train), 'train sequences')
print(len(input_test), 'test sequences')

input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
print('input_train shape:', input_train.shape)
print('input_test shape:' , input_test.shape)

#training the model with embedding and simpleRNN layers
from keras.layers import Dense
model= Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(32))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['acc'])
history = model.fit(input_train, y_train, epochs = 10, batch_size = 128, validation_split = 0.2)

#plot results
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
#Simple RNN is too simple, does not learn long-term dependencies


#More advanced layers- LSTM and GRU

#using the LSTM layer in Keras
from keras.layers import LSTM

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer= 'rmsprop', loss = 'binary_crossentropy', metrics = ['acc'])
history = model.fit(input_train, y_train, epochs = 10, batch_size = 128, validation_split = 0.2)
#achieve 88% validation accuracy
#could have performed better w/ regularization, tuning of hyperparameters such as the embeddings dimensionality or the LSTM output dimensionality
#particularly useful for question-answering and machine translation
