#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 20:48:56 2021

@author: willsteijn
"""

#from raw text to word embeddings
import os
imdb_dir = '/Users/Will/Documents/imdb/aclImdb'
train_dir = os.path.join(imdb_dir, 'train')

labels = []
texts = []

#processing the labels of the raw IMDB data
for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname), encoding = 'utf-8')
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)


                
#tokenizing the data - restrict the training data to the first 200 samples
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import numpy as np

maxlen = 100 #cuts review off after 100 words
training_samples = 200 #trains on 200 samples
validation_samples = 10000 #validates on 10,0000
max_words = 10000 #considers only the top 10,000 words

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index

print('Found %s unique tokens.' %len(word_index))
#88582 unique tokens
data = pad_sequences(sequences, maxlen = maxlen)

labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

#split the data into a training set and a validation set, but first shuffle the data, b/c we are starting with data in which samples are ordered (all neg first, the all positive second)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]

#download the GloVe word embeddings
#parsing the GloVe word-embeddigngs file
#https://nlp.stanford.edu/projects/glove
glove_dir = '/Users/Will/Documents/imdb/'
embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'), encoding = 'utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype = 'float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))
#found 400000 word vectors


#build an embedding matrix that can be loaded into an Embedding layer
#must be matrix of shape (max_words, embedding_dim) where each entry i contains the embedding_dim-dimensional vector for the word of index i in the reference word index (built  during tokenization)
#index 0 does not stand for any word or token - its a placeholder

#preparing the GloVe word-embeddings matrix
embedding_dim = 100

embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector #words not found in the embedding index will be all zeros

#model definition
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length = maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.summary()

#load the GloVe matrix into the Embedding layer, the first layer in th emodel
#loading pretrainied word embeddings into the embedding layer
model.layers[0].set_weights([embedding_matrix])
#freeze the embedded layer
model.layers[0].trainable = False

#Model training and evaluation
model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['acc'])
history = model.fit(x_train, y_train, epochs = 10, batch_size = 32, validation_data = (x_val, y_val))
#final val_acc = .5446
model.save_weights('pre_trained_glove_model.h5')

#plotting the results
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

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

#model quickly starts overfitting, which is unsurprising given the small number of training samples
#validation accuracy has high variation for the same reason


#can also train the model without loading the pretrained word embeddings and without freezing the embedding layer
#then, youd learn a task-specific embedding of the input tokens, which is more powerful when there is a lot of data available

#train the same model w/o pretrained embeddings
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
history = model.fit(x_train, y_train,epochs=10,batch_size=32,validation_data=(x_val, y_val))
#val acc worse than with pretrained embedding b/c not enough data

#evaluating the model on the test set

#tokenize the data of the test set
test_dir = os.path.join(imdb_dir, 'test')
labels = []
texts = []
for label_type in ['neg', 'pos']:
    dir_name = os.path.join(test_dir, label_type)
    for fname in sorted(os.listdir(dir_name)):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname), encoding = 'utf-8')
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)
sequences = tokenizer.texts_to_sequences(texts)
x_test = pad_sequences(sequences, maxlen=maxlen)
y_test = np.asarray(labels)

#evaluate the model on the test set
model.load_weights('pre_trained_glove_model.h5')
model.evaluate(x_test, y_test)
#accuracy of 55%