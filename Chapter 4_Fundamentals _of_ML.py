# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 14:30:55 2020

@author: Will
"""

#WAYS TO SEPARATE VALIDATION SET FROM TRAINING SET

#hold out validation

import numpy as np
num_validation_samples = 10000
np.random.shuffle(data) #shuffle the data 

#define the validation set
validation_data = data[:num_validation_samples]
data = data[num_validation_samples:]

#defines the training set
training_data = data[:]

#trains a model on the training data and evaluates it on the validation data
model = get_model()
model.train(training_data)
validation_score = model.evaluate(validation_data)

#tune the model, retrain it, evaluate, tune again...

#once hyperparams are tuned, train final model from scratch on all non-test data available
model = get_model()
model.train(np.concatenate([training_data, validation_data]))
test_score = model.evaluate(test_data)

#k-fold validation
k = 4
num_validation_samples = len(data)//k

np.random.shuffle(data)

validation_scores = []
for fold in range(k):
    #selects the validation-data partition
    validation_data = data[num_validation_samples * fold: num_validation_samples * (fold + 1)]
    #uses the remainder of the data as training data -note that the + operatior is list concatenation, not summation
    training_data = data[:num_validation_samples * fold] + data[num_validation_samples * (fold + 1):]
    #create a brand-new instance of the model (untrained)
    model = get_model()
    model.train(training_data)
    validation_score = model.evaluate(validation_data)
    validation_scores.append(validation_score)

#validation score is the average of the validation scores of the k folds    
validation_score = np.average(validation_scores)

#trains the findal model on all non-test data available
model = get_model()
model.train(data)
test_score = model.evaluate(test_data)


#iterated k-fold validation with shuffling
#use when you have relatively little data available
#consists of applying K-fold validation multiple times, shuffling the data every time before splitting it K ways
#very computationally expensive - end up training and evaluating P x K models

#feature vectorization
#normalize each feature independently to have a mean of 0 nand a std of 1
x -= x.mean(axis=0)
x /= x.std(axis=0)


#REGULARIZATION

#adding L2 weight regularization to the model
from keras import regularizers
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, kernel_regularizer = regularizers.l2(0.001), activation = 'relu', input_shape = (10000,)))
model.add(layers.Dense(16, kernel_regularizer = regularizers.l2(0.001), activation = 'relu', input_shape = (10000,)))
model.add(layers.Dense(1,activation = 'sigmoid'))
#l2(0.001) means every coefficient in the weight matrix of the layer will add 0.001*weight_coefficient_value to the toal loss of the network
#this penalty is only added at training time, so the loss for this network will be higher at training than test

#alernativey weight regularizers
from keras import regularizers
regularizers.l1(0.001) #L1 regularization
regularizers.l1_l2(l1=.001, l2=.001) #simultaneous L1 and L2 regularization


#adding drop out
#dropout rate is the fraction of the features that are zeroed out; usually between 0.2 and 0.5
#at test time the layer's output values are scaled down by a factor equal to the dropout rate

#at training time
layer_output *= np.random.randint(0, high = 2, size= layer_output.shape) #at training time, drops out 50% of the units in the output
#at test time
layer_output *= 0.5

#OR
#both at training time
layer_output *= np.random.randint(0, high = 2, size = layer_output.shape)
layer_output /= 0.5

#in keras, can indtroduce dropout in a network via the dropout layer which is applied to the output of the layer right before it
model.add(layers.Dropout(0.5))

#adding dropout to the imdb network (use with code from Chapter 3, Binary Classification)
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

#Once you’ve developed a satisfactory model configuration, you can train your final
#production model on all the available data (training and validation) and evaluate it
#one last time on the test set. If it turns out that performance on the test set is significantly
#worse than the performance measured on the validation data, this may mean
#either that your validation procedure wasn’t reliable after all, or that you began overfitting
#to the validation data while tuning the parameters of the model. In this case,
#you may want to switch to a more reliable evaluation protocol (such as iterated K-fold
#validation).




















