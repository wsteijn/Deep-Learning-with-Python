#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 20:48:28 2021

@author: willsteijn
"""

#Word Embeddings
#can be obtained in 2 ways:
#   1. learn word embeddingd jointly w/ the main task - start w/ random word vectors and learn word vectors in the same was as you learn the weights of a neural net
#   2. load into model the word embeddings that were precomputed = pretrained word embeddings

#learning word embeddings with the embedded layer
#instantiating an embedding layer
from keras.layers import Embedding
#embedding layer takes at least two arguments: number of possible tokens, and dimensionality of the embeddings
embedding_layer = Embedding(1000, 64)
#embedding layer maps integer indices to dense vectors
#takes as input a 2D tensor of integers, shape (samples, sequence_length)
#all sequences in a batch must have the same length - truncate long ones and pad short ones w/ zeros
#layer returns a 3D floating-point tensor of shape (samples, sequence_length, embedding_dimensionality)


#IMDB movie review example
#prepare the data - restrict movie reviews to top 10,000 most common words and cut off reviews after only 20 words
#network will learn 8 dimensional embeddings for each of the 10,000 words
#integer sequence will be turned into ebedded sequences, then flattened to 2D, then fed to a single Dense layer for classficiation
from keras.datasets import imdb
from keras import preprocessing

max_features = 10000 # number of words to consider as features
maxlen = 20 #cut off text after this number of words (among the max_features most common words)

#load data as lists of integers
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features,  skip_top=0, maxlen = maxlen, start_char=1, oov_char=2, index_from=3)
#turn the lists of integers into a 2D integer tensor of shape (samples, maxlen)
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

#using an embedding layer and classifier on imdb dataset
from keras.models import Sequential
from keras.layers import Flatten, Dense
model = Sequential()
model.add(Embedding(10000, 8, input_length=maxlen))#specifies the max input length to the embedidng layer so it can later be flattened 
model.add(Flatten())#flattens 3D tensor to 2D
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()
history = model.fit(x_train, y_train,epochs=10,batch_size=32,validation_split=0.2)

#using pretrained word embeddings
#use when you have so little training data available, that you cant use your data alone to learn an appropriate task-specific embedding
