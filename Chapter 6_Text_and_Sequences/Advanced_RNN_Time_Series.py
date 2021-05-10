#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 20:54:07 2021

@author: willsteijn
"""

#Advanced use of recurrent neural networks - time series data

#temperature-forecasting problem

#inspecting the data of the Jena weather dataset
import os

data_dir = 'C:/Users/Will/Downloads/jena_climate'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

f = open(fname)
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

print(header)
print(len(lines))
#outputs a count of 420,551 line of data - each line is a timestep

#parsing the data into numpy array
import numpy as np

float_data = np.zeros((len(lines), len(header)-1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values

#plot the temperature timeseries data
from matplotlib import pyplot as plt
temp = float_data[:,1] # tempearture (in degress Celsius)
plt.plot(range(len(temp)),temp)
#plot the frist 10 days of temperature in timeseries
plt.plot(range(1440), temp[:1440]) #temps are taken every 10 minutes, so 1440 temps are 10 days

#preparing the data
#in this data, a timestep = 10 minutes. all the following numbers are numbers of timesteps
lookback = 1440 #observations will go back 5 days
step = 6 #observations will be sampled at one data point per hour
delay = 144 #targets will be 24 hours in the future
batch_size = 128

#preprocess the data ro a format a neural network can ingest
#normalizing the data - normalize each timeseries independently so they all take small values on a similar scale
#write a python generator that takes the current array of float data and yields batches of data from the recent past, along with a target temp in the future. generate samples on the fly using the original data

#use the first 200,000 timesteps as training data
#normalizing
mean = float_data[:2000000].mean(axis = 0)
float_data -= mean
std = float_data[:200000].std(axis = 0)
float_data /= std
#generator
#takes the following arguments
#   data - original array of floating point data that was just normalized
#   lookback - how many timesteps back the input data should go
#   delay - how many timesteps in the future the target should be
#   min_index and max_index - indices in the data array that delimit which timesteps to draw from (useful for keeping a segment of the data for validation and another for testing)
#   shuffle - whether to shuffle the samples or draw them in chronological order
#   batch_size - the number of samples per batch
#   step = the period, in timesteps, at which you sample data (set it to 6 to draw one data point every hour)

def generator(data, lookback, delay, min_index, max_index,shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        samples = np.zeros((len(rows),lookback // step,data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets
        #generater to yield backwards-sequenced samples: yield samples[:, ::-1, :], targets 
        
#prepare training, validation, and test generators
#training will look at first 200,000 timesteps, validation looks at next 100000, test looks at remainder
train_gen = generator(float_data, lookback = lookback, delay = delay, min_index = 0, max_index = 200000, shuffle = True, step = step, batch_size = batch_size)
val_gen = generator(float_data, lookback = lookback, delay = delay, min_index = 200001, max_index = 300000, step = step, batch_size = batch_size)
test_gen = generator(float_data, lookback = lookback, delay = delay, min_index = 300001, max_index = None, step = step, batch_size = batch_size)
val_steps = (300000 - 200001 - lookback) #how  many steps to draw from val_gen in order to see the entire validation set
test_steps = (len(float_data) - 300001 - lookback)#how  many steps to draw from test_gen in order to see the entire test set

#set baseline to beat - predict that the temp 24 hours from now will be equal to the temp right now
#evaluate baseline apporach using MAE
def evaluate_naive_method():
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    print(np.mean(batch_maes))
evaluate_naive_method()
#gives MAE of 0.29 - multiply by temperature_std degrees gives 2.57 degrees averge error

#training and evaluating a densely connected model
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.Flatten(input_shape = (lookback //step, float_data.shape[-1])))
model.add(layers.Dense(32, activation = 'relu'))
model.add(layers.Dense(1))
model.compile(optimizer = RMSprop(), loss = 'mae')
history = model.fit_generator(train_gen, steps_per_epoch = 500, epochs = 20, validation_data = val_gen, validation_steps = val_steps)

#plotting the results
import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
#validation loss varies greatly in each epoch, not great

#now look at the data without Flattening it first - recurrent-sequence processing model
#use a GRU layer - Gated Recurrent Unit
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.GRU(32, input_shape = (None, float_data.shape[-1])))
model.add(layers.Dense(1))

model.compile(optimizer = RMSprop(), loss = 'mae')
history = model.fit_generator(train_gen, steps_per_epoch = 500, epochs = 20, validation_data = val_gen, validation_steps = val_steps//batch_size)

import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


#add dropout to fight overfitting
#the same pattern of dropped units is applied at every timestep and a temporally constant pattern of dropout units is applied to the inner recurrent activations of the layer
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.GRU(32,dropout=0.2,recurrent_dropout=0.2,input_shape=(None, float_data.shape[-1])))
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,steps_per_epoch=500,epochs=40,validation_data=val_gen,validation_steps=val_steps)


#validation curve shows more stable evaluation scores
#but still the best scores arent much lower than previous

#increase performance by increasing the capacity of the network

#training and evaluating a dropout-regularized stacked GRU model
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
model = Sequential()
model.add(layers.GRU(32,dropout=0.1,recurrent_dropout=0.5,return_sequences=True,input_shape=(None, float_data.shape[-1])))
model.add(layers.GRU(64, activation='relu',dropout=0.1,recurrent_dropout=0.5))
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,steps_per_epoch=500,epochs=40,validation_data=val_gen,validation_steps=val_steps)

