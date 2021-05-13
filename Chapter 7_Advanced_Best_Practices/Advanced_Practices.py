#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 17:24:09 2021

@author: willsteijn
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 11:11:03 2020

@author: Will
"""

#Functional API implementation of a two-input question-answering model
from keras.models import Model
from keras import layers
from keras import Input

text_vocabulary_size = 10000
question_vocabulary_size = 10000
answer_vocabulary_size = 500

#text input isa variable-length sequence of integers
text_input= Input(shape=(None,), dtype = 'int32', name = 'text')

#embed the inputs intoa sequence of vectors of size 64
embedded_text =layers.Embedding(text_vocabulary_size,64)(text_input)

#encode the vectors in a single vector via an LSTM
encoded_text = layers.LSTM(32)(embedded_text)

#do the same (with different layer instances) for the question
question_input = Input(shape=(None,), dtype = 'int32', name = 'question')
embedded_question = layers.Embedding(question_vocabulary_size, 32)(question_input)
encoded_question = layers.LSTM(16)(embedded_question)

#concatenate the encoded question and the encoded text
concatenated = layers.concatenate([encoded_text, encoded_question], axis =-1)

#add a softmax classifier on top
answer = layers.Dense(answer_vocabulary_size, activation = 'softmax')(concatenated)

#at model instantiation, you specify the two inputs and the output
model = Model([text_input, question_input], answer)
model.compile(optimizer='rmsprop', loss = 'categorical_crossentropy', metrics = ['acc'])

#how to train two input model
#two possible APIs: feed the model a list of Numpy arrays as inputs, or can feed it a dictionary that maps input names to Numpy arrays
#second option is available only if you give names ot inputs

#feeding data to a multi-input model
import numpy as np

num_samples = 1000
max_length =100

#generate dummy Numpy data
text = np.random.randint(1, text_vocabulary_size, size =(num_samples, max_length))
question = np.random.randint(1, question_vocabulary_size, size = (num_samples, max_length))
#answers are one-hot encoded, not integers
answers = np.random.randint(0,1,size = (num_samples, answer_vocabulary_size))

#fit model using a list of inputs
model.fit([text,question], answers, epochs = 10, batch_size = 128)

#fit model using a dcitonary of inputs(only if inputs are named)
model.fit({'text': text, 'question': question}, answers, epochs = 10, batch_size =128)


#Multi-output Models
#functional API implementation of a three-output model
from keras import layers
from keras import Input
from keras.models import Model

vocabulary_size = 50000
num_income_groups = 10

posts_input = Input(shape=(None,), dtype='int32', name='posts')
embedded_posts = layers.Embedding(256, vocabulary_size)(posts_input)
x = layers.Conv1D(128, 5, activation='relu')(embedded_posts)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(128, activation='relu')(x)

#output layers are given names
age_prediction = layers.Dense(1, name='age')(x)
income_prediction = layers.Dense(num_income_groups,activation='softmax',name='income')(x)
gender_prediction = layers.Dense(1, activation='sigmoid', name='gender')(x)

model = Model(posts_input,[age_prediction, income_prediction, gender_prediction])

#to train the model, must combine different losses for the different outputs into one loss
#compilation options of a multi-output model: multiple losses
model.compile(optimizer='rmsprop', loss = ['mse', 'categorical_crossentropy', 'binary_crossentropy'])
model.compile(optimizer='rmsprop', loss = {'age':'mse', 'income':'categorical_crossentropy', 'gender':'binary_crossentropy'})
#very imbalaned loss contributions will cause the model representations to be optimized preferentially for the task with the largest individual loss, at the expense of the other tasks
#to rememdy, can assign different importance to the loss values in their contribution to the final loss

#loss weighting
model.compile(optimizer='rmsprop', loss = ['mse', 'categorical_crossentropy', 'binary_crossentropy'], loss_weights = [0.25, 1., 10.])

#feeding data to a multi-output model
#age_targets, income_targets and gender_targets are assumed to be Numpy arrays
model.fit(posts, [age_targets, income_targets, gender_targets], epochs =10, batch_size =64)


#Layer Weight SHaring
#have ability to reuse a layer instanceseveral times - reuse the same weights withevery call
#allows to build models with shared branches - share the same representations and learn representaitons simultaneously for different set of inputs
#shared LSTM -want to process two inputs with a single LSTM layer

from keras import layers
from keras import Input
from keras.models import Model

#instantiate a single LSTM layer, once
lstm = layers.LSTM(32)

#build the left brance of the model: inputsare variable length sequences of vector size 128
left_input = Input(shape = (None, 128))
left_output = lstm(left_input)

#buuild the right branch  of the model: when you call an existing layer instance, you reuse its weights
right_input = Input(shape = (None, 128))
right_output = lstm(right_input)

#build the classifier on top
merged = layers.concatenate([left_output, right_output], axis = -1)
predictions = layers.Dense(1, activation = 'sigmoid')(merged)

#instantiate and train the model: when the model is trained, the weights of the LSTM layer are updated based on both inputs
model = Model([left_input, right_input], predictions)
model.fit([left_data, right_data], targets)


#Models as Layers
#you can call a model on an input tensor and retrieve an output tensor
y = model(x)

#if the model has multiple input tensors and multiple output tensors, it can be called with a list of tensors
y1, y2 = model([x1, x2])
#when you call a model instance, you reusue the weights of the model

#practical example: a vision model that uses a dual camera as its input: two parallel cameras, a few centimeters apart
#such a model can perceive depth
#shouldnt  need two independentmodels to extract visual features from the left and right cameras before merging the two feeds
#can be shared across the two inputs
from keras import layers
from keras import applications
from keras import Input
#the base image-processing modelis the Xception netowrk(convolutional base only)
xception_base = applications.Xception(weights=None,include_top=False)

#inputs are 250 x 250 RGB images
left_input = Input(shape=(250, 250, 3))
right_input = Input(shape=(250, 250, 3))

#call the same vision model twice
left_features = xception_base(left_input)
right_input = xception_base(right_input)

#merged features contain information from the right visual feed and the left visual feed
merged_features = layers.concatenate([left_features, right_input], axis=-1)


#Improving the model.fit call

#callbacks

#the modelcheckpoint and earlystopping callbacks
#EarlyStopping callback can be used to interrupt training once a target metric stops improving for a fixed number of epochs- allows you to stop as soon as overfitting starts
#ModelCheckpoint lets you continually save the model during training, andoptionally only the current best model so far
import keras

#early stopping interrupts training when improvement stops, monitors validation accuracy, interrupts training when accuracy has stopped improving for more than one epoch
#modelcheckpoint saves the current weights after each epoch, wont overwrite the model file unless val_loss has improved (keeps best model seen during training)
callbacks_list = [keras.callbacks.EarlyStopping(monitor='acc',patience=1,),keras.callbacks.ModelCheckpoint(filepath='my_model.h5',monitor='val_loss',save_best_only=True,)]
#monitor accuracy, so it should be part of the model's metrics that are set in compile
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
#because the callback will monitor validation loss and validation accuracy, need to pass validation_data to the call to fit
model.fit(x, y,epochs=10,batch_size=32,callbacks=callbacks_list,validation_data=(x_val, y_val))

#reduceLRonPlateau callback
#can use this callback to reduce the learning rate when the validation loss has stopped improving - effective strategy to get out of a local minima during training
callbacks_list = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=10)]
model.fit(x, y,epochs=10,batch_size=32,callbacks=callbacks_list,validation_data=(x_val, y_val))



#TensorBoard - Visualizing Results

#text-classification model to use with TensorBoard
import keras
from keras import layers
from keras.datasets import imdb
from keras.preprocessing import sequence
max_features = 2000
max_len = 500
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)
model = keras.models.Sequential()
model.add(layers.Embedding(max_features, 128,input_length=max_len,name='embed'))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))
model.summary()
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])

#create a directory for TensorBoard log files
mkdir my_log_dir

#training the model with a TensorBoard callback
#record activation histogram every 1 epoch
#record embedding data every 1 epoch
callbacks = [keras.callbacks.TensorBoard(log_dir='my_log_dir',histogram_freq=1,embeddings_freq=1,)]
history = model.fit(x_train, y_train,epochs=20,batch_size=128,validation_split=0.2,callbacks=callbacks)

%load_ext tensorboard

%tensorboard --logdir my_log_dir
#then go to http://localhost:6006 to see results

%pip install pydot
from keras.utils import plot_model
plot_model(model, show_shapes=True, to_file='model.png')

#Depthwise Separable Convolution
from keras.models import Sequential, Model
from keras import layers
height = 64
width = 64
channels = 3
num_classes = 10
model = Sequential()
model.add(layers.SeparableConv2D(32, 3,activation='relu',input_shape=(height, width, channels,)))
model.add(layers.SeparableConv2D(64, 3, activation='relu'))
model.add(layers.MaxPooling2D(2))
model.add(layers.SeparableConv2D(64, 3, activation='relu'))
model.add(layers.SeparableConv2D(128, 3, activation='relu'))
model.add(layers.MaxPooling2D(2))
model.add(layers.SeparableConv2D(64, 3, activation='relu'))
model.add(layers.SeparableConv2D(128, 3, activation='relu'))
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

#Hyperparameter optimization
#The process of optimizing hyperparameters typically looks like this:
#1 Choose a set of hyperparameters (automatically).
#2 Build the corresponding model.
#3 Fit it to your training data, and measure the final performance on the validation
#data.
#4 Choose the next set of hyperparameters to try (automatically).
#5 Repeat.
#6 Eventually, measure performance on your test data.
#The key to this process is the algorithm that uses this history of validation performance,
#given various sets of hyperparameters, to choose the next set of hyperparameters
#to evaluate. Many different techniques are possible: Bayesian optimization,
#genetic algorithms, simple random search, and so on.
#Be mindful of validation-set overfitting
#Hyperparameter optimization toosl:
#Hyperopt
#Hyperas

#Model Ensembeling
#A smarter way to ensemble classifiers is to do a weighted average, where the
#weights are learned on the validation data—typically, the better classifiers are given a
#higher weight, and the worse classifiers are given a lower weight. To search for a good
#set of ensembling weights, you can use random search or a simple optimization algorithm
#such as Nelder-Mead:
#preds_a = model_a.predict(x_val)
#preds_b = model_b.predict(x_val)
#preds_c = model_c.predict(x_val)
#preds_d = model_d.predict(x_val)
#final_preds = 0.5 * preds_a + 0.25 * preds_b + 0.1 * preds_c + 0.15 * preds_d
#the key to ensembling is the diversity of the set of classifiers
#works well - the use of an ensemble of tree-based methods (such as random
#forests or gradient-boosted trees) and deep neural networks
