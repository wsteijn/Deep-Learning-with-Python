#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  8 14:21:25 2021

@author: willsteijn
"""

#One-hot encoding of words and characters
#associates a unique vector integer index with every word and then turns this integer index i into a binary vector of size N (the size of the vocabulary)
#the vector is all zeros except for the ith entry, which is 1

#word-level one-hot encoding toy example
import numpy as np
#initial data: one entry per sample(a sample is a sentence in this example, but could be an entire document)
samples = ['The cat sat on the mat.', 'The dog ate my homework.']
#build index of all tokens in the data
token_index = {}
for sample in samples:
    #tokenize teh sample via split  method (should also strip punctuation and special characters from the samples)
    for word in sample.split():
        if word not in token_index:
            #assign a unique index to each unique word. dont attribute index 0 to anything
            token_index[word] = len(token_index) + 1
    
#vectorize the samples - only consider the first max_length words in each sample
max_length = 10
#make place to store the results
results = np.zeros(shape = (len(samples),max_length, max(token_index.values()) + 1))
for i, samples in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        results[i, j, index] = 1.

#character-level one-hot encoding toy example
import string
samples = ['The cat sat on the mat.', 'The dog ate my homework.']
#define all printable ASCII characters
characters = string.printable
token_index = dict(zip(range(1, len(characters) +1), characters))
max_length = 50
results = np.zeros((len(samples), max_length, max(token_index.keys()) + 1))
for i, sample in enumerate(samples):
    for j, character in enumerate(sample):
        index = token_index.get(character)
        results[i, j, index] = 1.

#using Keras for word-level one-hot encoding
#Keras takes care of stripping special characters from strings and only taking into account the N most common words in the dataset
from keras.preprocessing.text import Tokenizer
samples = ['The cat sat on the mat.', 'The dog ate my homework.']

#create a tokenizer, configured to take into account only the 1000 most common words
tokenizer = Tokenizer(num_words = 1000)
#build the word index
tokenizer.fit_on_texts(samples)
#turn strings into lists of integer indices
sequences = tokenizer.texts_to_sequences(samples)
#can also directly get the one-hot binary representations. vectorization modes other than one-hot encoding are supported by this tokenizer
one_hot_results = tokenizer.texts_to_matrix(samples, mode = 'binary')
#recover word index that was computed
word_index = tokenizer.word_index
print('Found number of unique tokens.', len(word_index))

#one-hot hashing trick
#use when the number of unique words in vocabulary is too large to handle explicitly
#words are hashed into vectors of fixed size
#does away with maintaining an explicit word index - saves memory and allows online encoding of the data
#susceptible to hash collisions - likelihood of has collisions decreases when the dimensionality of the hashing space is much larger than the total number of unique tokens being generated

#word-level one-hot encoding with hashing trick toy example
samples = ['The cat sat on the mat.', 'The dog ate my homework.']
#store words as vectors of size 1,000. If you have close to 1000 words (or more), there will be many hash collisions which will decrease the accuracy of the encoding method
dimensionality = 1000 
max_length = 10
results = np.zeros((len(samples), max_length, dimensionality))
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = abs(hash(word)) % dimensionality #hashes the word into a random integer index between 0 and 1000
        results[i, j, index] = 1.

