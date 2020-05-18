# -*- coding: utf-8 -*-
"""
Created on Wed May 13 15:28:37 2020

@author: mauri
"""
import scipy.io as sio
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

# import used files
import DVH_functions

#%% Loading the data
# load input features and create (1000,6) matrix of input features
mat_abscissas_file = 'C:/Users/mauri/Documents/BEP/Data/Patient_2/abscissas.mat'
mat_vaeSC_file = 'C:/Users/mauri/Documents/BEP/Data/Patient_2/vaeSCv7.mat'
abscissas_file = sio.loadmat(mat_abscissas_file)
vaeSC_file = sio.loadmat(mat_vaeSC_file)
abscissas = abscissas_file['abscissas'] # contains first 5 input features
StartCalc = vaeSC_file['startCalc'] # contains 6th input feature

input_features = np.column_stack((abscissas,StartCalc)) # put all input features in one matrix

# load output dose distributions and DVHs

mat_output_file = 'C:/Users/mauri/Documents/BEP/Data/Patient_2/output.mat'
output_file = sio.loadmat(mat_output_file)
Dose = output_file['Dose'] # contains dose data
dvh = output_file['dvh'] # contains dvh data

#%% random shuffling of the samples. The rows of dvh and input_features are shuffeld randomly but in the same way.
x = input_features
y = dvh

a = np.array(x)
b = np.array(y)

indices = np.arange(a.shape[0])
np.random.shuffle(indices)

input_features = a[indices]
dvh = b[indices]

#%% Preparing the data
# splitting the dose and dvh data in train and unseen test data
training_samples = 950 # the number of examples used to train the model
X_train = input_features[:training_samples] 
X_test = input_features[training_samples:]
Dose_train = Dose[: training_samples]
Dose_test = Dose[training_samples :]
dvh_train = dvh[:training_samples]
dvh_test = dvh[training_samples :]

# standardizing the data
"""scaler_X.mean_ gives the mean values, scaler_X.scale_ gives the std values"""
scaler_X = preprocessing.StandardScaler().fit(X_train) #scaler for the input data
scaler_dvh = preprocessing.StandardScaler().fit(dvh_train) #scaler for the ouput dvh data
X_train_scaled = scaler_X.transform(X_train) #scale input training data
dvh_train_scaled = scaler_dvh.transform(dvh_train) #scale output training data
X_test_scaled = scaler_X.transform(X_test) # scale input test data


# function to retrieve output Y from scaled output Y
""""Y_scaled is the scaled data from which the unscaled data is to be retrieved
    scaler_Y is the scaler that is used to retrieve the data from Y_scaled"""
def scale_output(Y_scaled, scaler_Y):
    Y = Y_scaled*scaler_Y.scale_ + scaler_Y.mean_
    return Y


#%% mean as an estimator

test = scaler_dvh.mean_
DVH_functions.dvh_eval(dvh_test, test)

#%% Not normalized model
def get_callbacks_not_normed(name):
  return [
    tfdocs.modeling.EpochDots(),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=100),
    #keras.callbacks.TensorBoard(logdir/name),
  ]
def compile_and_fit_not_normed(model, name, max_epochs=1000):
    
  optimizer = tf.keras.optimizers.Adam(0.1)
  loss=tf.keras.losses.MeanSquaredError()
  model.compile(loss=loss,
                optimizer=optimizer,
                metrics=['mae', 'mse'])

  model.summary()

  history = model.fit(
    X_train,
    dvh_train,
    epochs=max_epochs,
    validation_split = 0.2,
    callbacks=get_callbacks_not_normed(name),
    verbose=0)
  return history
not_normed_model = keras.Sequential([
    layers.Dense(30, activation='relu', input_shape=[X_train.shape[1]]),
    layers.Dense(100, activation='relu'),
    layers.Dense(100),
    
  ])
histories_normed = {}
histories_normed['not_normed'] = compile_and_fit_not_normed(not_normed_model, 'model/not_normed')
not_normed = not_normed_model.predict(X_test)
DVH_functions.dvh_eval(dvh_test, not_normed)
#DVH_functions.plot_dvh(dvh_test, not_normed)
#%% Normalized model
def get_callbacks_normed(name):
  return [
    tfdocs.modeling.EpochDots(),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=100),
    #keras.callbacks.TensorBoard(logdir/name),
  ]
def compile_and_fit_normed(model, name, max_epochs=1000):
    
  optimizer = tf.keras.optimizers.Adam(0.1)
  loss=tf.keras.losses.MeanSquaredError()
  model.compile(loss=loss,
                optimizer=optimizer,
                metrics=['mae', 'mse'])

  model.summary()

  history = model.fit(
    X_train_scaled,
    dvh_train_scaled,
    epochs=max_epochs,
    validation_split = 0.2,
    callbacks=get_callbacks_normed(name),
    verbose=0)
  return history
normed_model = keras.Sequential([
    layers.Dense(30, activation='relu', input_shape=[X_train_scaled.shape[1]]),
    layers.Dense(100, activation='relu'),
    layers.Dense(100),
    
  ])
histories_normalization = {}
histories_normalization = compile_and_fit_normed(normed_model, 'model/normed')

normed = normed_model.predict(X_test_scaled)
normed_not_normed = scale_output(normed, scaler_dvh)
DVH_functions.dvh_eval(dvh_test, normed_not_normed)
#DVH_functions.plot_dvh(dvh_test, not_normed)

#%% Functions to compile and fit the model
def get_callbacks(name):
  return [
    tfdocs.modeling.EpochDots(),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=60),
    #keras.callbacks.TensorBoard(logdir/name),
  ]
def compile_and_fit(model, name, max_epochs=1000):
  optimizer = tf.keras.optimizers.Adam(0.01)
  loss=tf.keras.losses.MeanSquaredError()
  model.compile(loss=loss,
                optimizer=optimizer,
                metrics=['mae', 'mse'])

  model.summary()

  history = model.fit(
    X_train_scaled,
    dvh_train_scaled,
    epochs=max_epochs,
    validation_split = 0.2,
    callbacks=get_callbacks(name),
    verbose=0)
  return history

#%%
histories = {}

#%% simple 4, 8, 12, 16, 20 layer models
"""Test to see how the performance depends on the number of layers
doubling the amount of hidden neurons each layers with the output as maximum

conclusion: 
"""
histories = {}
# 4 layer model simple
layers4_model = keras.Sequential([
    layers.Dense(12, activation='relu', input_shape=[X_train_scaled.shape[1]]),
    layers.Dense(24, activation='relu'),
    layers.Dense(48, activation='relu'),
    layers.Dense(96, activation='relu'),
    layers.Dense(100),
    
  ])

histories['layers4'] = compile_and_fit(layers4_model, 'model/layers4')

layers4_scaled = layers4_model.predict(X_test_scaled)
layers4 = scale_output(layers4_scaled, scaler_dvh)

# 8 layer model simple
layers8_model = keras.Sequential([
    layers.Dense(12, activation='relu', input_shape=[X_train_scaled.shape[1]]),
    layers.Dense(24, activation='relu'),
    layers.Dense(48, activation='relu'),
    layers.Dense(96, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100),
    
  ])

histories['layers8'] = compile_and_fit(layers8_model, 'model/layers8')

layers8_scaled = layers8_model.predict(X_test_scaled)
layers8 = scale_output(layers8_scaled, scaler_dvh)

# 12 layer model simple
layers12_model = keras.Sequential([
    layers.Dense(12, activation='relu', input_shape=[X_train_scaled.shape[1]]),
    layers.Dense(24, activation='relu'),
    layers.Dense(48, activation='relu'),
    layers.Dense(96, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100),
    
  ])

histories['layers12'] = compile_and_fit(layers12_model, 'model/layers12')

layers12_scaled = layers12_model.predict(X_test_scaled)
layers12 = scale_output(layers12_scaled, scaler_dvh)

# 16 layer model simple
layers16_model = keras.Sequential([
    layers.Dense(12, activation='relu', input_shape=([X_train_scaled.shape[1]])),
    layers.Dense(24, activation='relu', kernel_initializer='glorot_uniform',bias_initializer='zeros'),
    layers.Dense(48, activation='relu'),
    layers.Dense(96, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100),
    
  ])

histories['layers16'] = compile_and_fit(layers16_model, 'model/layers16')

layers16_scaled = layers16_model.predict(X_test_scaled)
layers16 = scale_output(layers16_scaled, scaler_dvh)

# 20 layer model simple
layers20_model = keras.Sequential([
    layers.Dense(12, activation='relu', input_shape=([X_train_scaled.shape[1]])),
    layers.Dense(24, activation='relu', kernel_initializer='glorot_uniform',bias_initializer='zeros'),
    layers.Dense(48, activation='relu'),
    layers.Dense(96, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),  
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100),
    
  ])

histories['layers20'] = compile_and_fit(layers20_model, 'model/layers20')

layers20_scaled = layers20_model.predict(X_test_scaled)
layers20 = scale_output(layers20_scaled, scaler_dvh)

#commands to plot histograms with the minimal, mean and maximal relative errors and a table with the mean errors
#DVH_functions.dvh_eval(dvh_test, layers4)
#DVH_functions.dvh_eval(dvh_test, layers8)
#DVH_functions.dvh_eval(dvh_test, layers12)
#DVH_functions.dvh_eval(dvh_test, layers16)
#DVH_functions.dvh_eval(dvh_test, layers20)

# plot the train and validation mse as a funciton of the number of epochs
plt.figure(2)
plotter = tfdocs.plots.HistoryPlotter(metric = 'mse', smoothing_std=10)
plotter.plot(histories)

plt.ylim([0.9, 1.0])
plt.xlim([0, 25])
#%% #%% simple 4, 8, 12, 16, 20 layer models
"""Test to se what amount of hidden neurons works best 3 options tested
option 1: doubling the amount of hidden neurons each layers with the output as maximum
option 2: each layer the amount of hidden neurons of the previous layer x1.5 and /1.5 to the output
option 3: each layer the amount of hidden neurons of the previous layer x2.0 and /2.0 to the output

tested for 8 and 12 layer models.
Conclusion: option 1 seems to be the best"""
histories = {}
# 8 layer model simple
layers8_model = keras.Sequential([
    layers.Dense(12, activation='relu', input_shape=[X_train_scaled.shape[1]]),
    layers.Dense(24, activation='relu'),
    layers.Dense(48, activation='relu'),
    layers.Dense(96, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100),
    
  ])

histories['layers8'] = compile_and_fit(layers8_model, 'model/layers8')

layers8_scaled = layers8_model.predict(X_test_scaled)
layers8 = scale_output(layers8_scaled, scaler_dvh)

# 12 layer model simple
layers12_model = keras.Sequential([
    layers.Dense(12, activation='relu', input_shape=[X_train_scaled.shape[1]]),
    layers.Dense(24, activation='relu'),
    layers.Dense(48, activation='relu'),
    layers.Dense(96, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100),
    
  ])

histories['layers12'] = compile_and_fit(layers12_model, 'model/layers12')

layers12_scaled = layers12_model.predict(X_test_scaled)
layers12 = scale_output(layers12_scaled, scaler_dvh)

# 8 layer model simple each layer 1.5x more hidden neurons
layers8_15_model = keras.Sequential([
    layers.Dense(6, activation='relu', input_shape=[X_train_scaled.shape[1]]),
    layers.Dense(9, activation='relu'),
    layers.Dense(14, activation='relu'),
    layers.Dense(21, activation='relu'),
    layers.Dense(31, activation='relu'),
    layers.Dense(47, activation='relu'),
    layers.Dense(51, activation='relu'),
    layers.Dense(77, activation='relu'),
    layers.Dense(100),
    
  ])

histories['layers8_1.5'] = compile_and_fit(layers8_15_model, 'layers8_1.5')

layers8_15_scaled = layers8_15_model.predict(X_test_scaled)
layers8_15 = scale_output(layers8_15_scaled, scaler_dvh)

# 12 layer model simple each layer 1.5x more hidden neurons
layers12_15_model = keras.Sequential([
    layers.Dense(6, activation='relu', input_shape=[X_train_scaled.shape[1]]),
    layers.Dense(9, activation='relu'),
    layers.Dense(14, activation='relu'),
    layers.Dense(21, activation='relu'),
    layers.Dense(31, activation='relu'),
    layers.Dense(47, activation='relu'),
    layers.Dense(51, activation='relu'),
    layers.Dense(77, activation='relu'),
    layers.Dense(116, activation='relu'),
    layers.Dense(174, activation='relu'),
    layers.Dense(225, activation='relu'),
    layers.Dense(150, activation='relu'),
    layers.Dense(100),
    
  ])

histories['layers12_1.5'] = compile_and_fit(layers12_15_model, 'model/layers12_1.5')

layers12_15_scaled = layers12_15_model.predict(X_test_scaled)
layers12_15 = scale_output(layers12_15_scaled, scaler_dvh)

# 8 layer model simple each layer 1.5x more hidden neurons
layers8_20_model = keras.Sequential([
    layers.Dense(6, activation='relu', input_shape=[X_train_scaled.shape[1]]),
    layers.Dense(12, activation='relu'),
    layers.Dense(24, activation='relu'),
    layers.Dense(48, activation='relu'),
    layers.Dense(96, activation='relu'),
    layers.Dense(192, activation='relu'),
    layers.Dense(384, activation='relu'),
    layers.Dense(200, activation='relu'),
    layers.Dense(100),
    
  ])

histories['layers8_2.0'] = compile_and_fit(layers8_20_model, 'layers8_2.0')

layers8_20_scaled = layers8_20_model.predict(X_test_scaled)
layers8_20 = scale_output(layers8_20_scaled, scaler_dvh)

# 12 layer model simple each layer 2.0x more hidden neurons
layers12_20_model = keras.Sequential([
    layers.Dense(6, activation='relu', input_shape=[X_train_scaled.shape[1]]),
    layers.Dense(12, activation='relu'),
    layers.Dense(24, activation='relu'),
    layers.Dense(48, activation='relu'),
    layers.Dense(96, activation='relu'),
    layers.Dense(192, activation='relu'),
    layers.Dense(384, activation='relu'),
    layers.Dense(768, activation='relu'),
    layers.Dense(1536, activation='relu'),
    layers.Dense(800, activation='relu'),
    layers.Dense(400, activation='relu'),
    layers.Dense(200, activation='relu'),
    layers.Dense(100),
    
  ])

histories['layers12_2.0'] = compile_and_fit(layers12_20_model, 'model/layers12_2.0')

layers12_20_scaled = layers12_20_model.predict(X_test_scaled)
layers12_20 = scale_output(layers12_20_scaled, scaler_dvh)

#commands to plot histograms with the minimal, mean and maximal relative errors and a table with the mean errors
#DVH_functions.dvh_eval(dvh_test, layers8)
#DVH_functions.dvh_eval(dvh_test, layers8_15)
#DVH_functions.dvh_eval(dvh_test, layers8_20)
#DVH_functions.dvh_eval(dvh_test, layers12)
#DVH_functions.dvh_eval(dvh_test, layers12_15)
#DVH_functions.dvh_eval(dvh_test, layers12_20)

# plot the train and validation mse as a funciton of the number of epochs
plt.figure(3)
plotter = tfdocs.plots.HistoryPlotter(metric = 'mse', smoothing_std=10)
plotter.plot(histories)

plt.ylim([0.9, 1.0])
plt.xlim([0, 25])

#%% adding batchnormalization layers
"""test the influence of batchnormalization() layers, three options are tested:
    option 1: without batchnormalization layers
    option 2: a batchnormlayer after every second dense layer
    option 3: a batchnormlayer after every dense layer"""
histories = {}
# 8 layer model simple
layers8_model = keras.Sequential([
    layers.Dense(12, activation='relu', input_shape=[X_train_scaled.shape[1]]),
    layers.Dense(24, activation='relu'),
    layers.Dense(48, activation='relu'),
    layers.Dense(96, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100),
    
  ])

histories['layers8'] = compile_and_fit(layers8_model, 'model/layers8')

layers8_scaled = layers8_model.predict(X_test_scaled)
layers8 = scale_output(layers8_scaled, scaler_dvh)

# 8 layer model simple with a batchnormalization layer after every 2 dense layers
layers8_norm_model = keras.Sequential([
    layers.Dense(12, input_shape=[X_train_scaled.shape[1]]),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dense(24, activation='relu'),
    layers.Dense(48),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dense(96, activation='relu'),
    layers.Dense(100),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dense(100, activation='relu'),
    layers.Dense(100),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dense(100, activation='relu'),
    layers.Dense(100),
    
  ])

histories['layers8_norm'] = compile_and_fit(layers8_norm_model, 'model/layers8_norm')

layers8_norm_scaled = layers8_norm_model.predict(X_test_scaled)
layers8_norm = scale_output(layers8_norm_scaled, scaler_dvh)

# 12 layer model with a batchnormalization layer after every second dense layers
layers12_norm_model = keras.Sequential([
    layers.Dense(12, input_shape=[X_train_scaled.shape[1]]),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dense(24),
    layers.Activation("relu"),
    layers.Dense(48),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dense(96),
    layers.Activation("relu"),
    layers.Dense(100),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dense(100),
    layers.Activation("relu"),
    layers.Dense(100),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dense(100),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dense(100),
    layers.Activation("relu"),
    layers.Dense(100),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dense(100),
    layers.Activation("relu"),
    layers.Dense(100),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dense(100),
    
  ])

histories['layers12_norm'] = compile_and_fit(layers12_norm_model, 'model/layers12_norm')

layers12_norm_scaled = layers12_norm_model.predict(X_test_scaled)
layers12_norm = scale_output(layers12_norm_scaled, scaler_dvh)

#commands to plot histograms with the minimal, mean and maximal relative errors and a table with the mean errors
#DVH_functions.dvh_eval(dvh_test, layers8)
#DVH_functions.dvh_eval(dvh_test, layers8_norm)
#DVH_functions.dvh_eval(dvh_test, layers12_norm)

plt.figure(4)
plotter = tfdocs.plots.HistoryPlotter(metric = 'mse', smoothing_std=10)
plotter.plot(histories)

plt.ylim([0.6, 1.1])
plt.xlim([0, 100])

#%% adding dropout layers
"""test the influence of batchnormalization() layers, three options are tested:
    option 1: 8 layer model without dropout layers
    option 2: 8 layer model with a dropout after every second dense layer
    option 3: 12 layer model with a dropout after every dense layer"""
histories = {}
dropout = 0.15
# 8 layer model simple
layers8_model = keras.Sequential([
    layers.Dense(12, activation='relu', input_shape=[X_train_scaled.shape[1]]),
    layers.Dense(24, activation='relu'),
    layers.Dense(48, activation='relu'),
    layers.Dense(96, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100),
    
  ])

histories['layers8'] = compile_and_fit(layers8_model, 'model/layers8')

layers8_scaled = layers8_model.predict(X_test_scaled)
layers8 = scale_output(layers8_scaled, scaler_dvh)

# 8 layer model  with a dropout layer after every second dense layers
layers8_drop_model = keras.Sequential([
    layers.Dense(12, input_shape=[X_train_scaled.shape[1]]),
    layers.Activation("relu"),
    layers.Dropout(dropout),
    layers.Dense(24, activation='relu'),
    layers.Dense(48),
    layers.Activation("relu"),
    layers.Dropout(dropout),
    layers.Dense(96, activation='relu'),
    layers.Dense(100),
    layers.Activation("relu"),
    layers.Dropout(dropout),
    layers.Dense(100, activation='relu'),
    layers.Dense(100),
    layers.Activation("relu"),
    layers.Dropout(dropout),
    layers.Dense(100, activation='relu'),
    layers.Dense(100),
    
  ])

histories['layers8_drop'] = compile_and_fit(layers8_drop_model, 'model/layers8_drop')

layers8_drop_scaled = layers8_drop_model.predict(X_test_scaled)
layers8_drop = scale_output(layers8_drop_scaled, scaler_dvh)

# 12 layer model simple layer model simple with a dropout layer after every second dense layers
layers12_drop_model = keras.Sequential([
    layers.Dense(12, input_shape=[X_train_scaled.shape[1]]),
    layers.Activation("relu"),
    layers.Dropout(dropout),
    layers.Dense(24),
    layers.Activation("relu"),
    layers.Dense(48),
    layers.Activation("relu"),
    layers.Dropout(dropout),
    layers.Dense(96),
    layers.Activation("relu"),
    layers.Dense(100),
    layers.Activation("relu"),
    layers.Dropout(dropout),
    layers.Dense(100),
    layers.Activation("relu"),
    layers.Dense(100),
    layers.Activation("relu"),
    layers.Dropout(dropout),
    layers.Dense(100),
    layers.Activation("relu"),
    layers.Dropout(dropout),
    layers.Dense(100),
    layers.Activation("relu"),
    layers.Dense(100),
    layers.Activation("relu"),
    layers.Dropout(dropout),
    layers.Dense(100),
    layers.Activation("relu"),
    layers.Dense(100),
    layers.Activation("relu"),
    layers.Dropout(dropout),
    layers.Dense(100),
    
  ])

histories['layers12_drop'] = compile_and_fit(layers12_drop_model, 'model/layers12_drop')

layers12_drop_scaled = layers12_drop_model.predict(X_test_scaled)
layers12_drop = scale_output(layers12_drop_scaled, scaler_dvh)

#commands to plot histograms with the minimal, mean and maximal relative errors and a table with the mean errors
#DVH_functions.dvh_eval(dvh_test, layers8)
#DVH_functions.dvh_eval(dvh_test, layers8_drop)
#DVH_functions.dvh_eval(dvh_test, layers12_drop)

plt.figure(5)
plotter = tfdocs.plots.HistoryPlotter(metric = 'mse', smoothing_std=10)
plotter.plot(histories)

plt.ylim([0.6, 1.1])
plt.xlim([0, 100])

#%% batchnorm and dropout layers
histories = {}
dropout = 0.1
# 12 layer model simple with a dropout and batchnormalization layer after every second dense layers
layers8_drop_norm_model = keras.Sequential([
    layers.Dense(12, input_shape=[X_train_scaled.shape[1]]),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dropout(dropout),
    layers.Dense(24, activation='relu'),
    layers.Dense(48),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dropout(dropout),
    layers.Dense(96, activation='relu'),
    layers.Dense(100),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dropout(dropout),
    layers.Dense(100, activation='relu'),
    layers.Dense(100),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dropout(dropout),
    layers.Dense(100, activation='relu'),
    layers.Dense(100),
    
  ])

histories['layers8_drop_norm'] = compile_and_fit(layers8_drop_norm_model, 'model/layers8_drop_norm')

layers8_drop_norm_scaled = layers8_drop_norm_model.predict(X_test_scaled)
layers8_drop_norm = scale_output(layers8_drop_norm_scaled, scaler_dvh)

# 12 layer model simple layer model simple with a dropout and batchnormalization layer after every second dense layers
layers12_drop_norm_model = keras.Sequential([
    layers.Dense(12, input_shape=[X_train_scaled.shape[1]]),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dropout(dropout),
    layers.Dense(24),
    layers.Activation("relu"),
    layers.Dense(48),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dropout(dropout),
    layers.Dense(96),
    layers.Activation("relu"),
    layers.Dense(100),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dropout(dropout),
    layers.Dense(100),
    layers.Activation("relu"),
    layers.Dense(100),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dropout(dropout),
    layers.Dense(100),
    layers.Activation("relu"),
    layers.Dense(100),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dropout(dropout),
    layers.Dense(100),
    layers.Activation("relu"),
    layers.Dense(100),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dropout(dropout),
    layers.Dense(100),
    layers.Activation("relu"), 
    layers.Dense(100),
    
  ])

histories['layers12_drop_norm'] = compile_and_fit(layers12_drop_norm_model, 'model/layers12_drop_norm')

layers12_drop_norm_scaled = layers12_drop_norm_model.predict(X_test_scaled)
layers12_drop_norm = scale_output(layers12_drop_norm_scaled, scaler_dvh)

#commands to plot histograms with the minimal, mean and maximal relative errors and a table with the mean errors
#DVH_functions.dvh_eval(dvh_test, layers8_drop_norm)
#DVH_functions.dvh_eval(dvh_test, layers12_drop_norm)

plt.figure(6)
plotter = tfdocs.plots.HistoryPlotter(metric = 'mse', smoothing_std=10)
plotter.plot(histories)

plt.ylim([0.6, 1.1])
plt.xlim([0, 500])

#%% Am I approximating the mean or the true values?
""""""
bins = 20
D2_predicted = layers12_drop_norm[:,49]
D2_test = dvh_test[:,49]
#plt.hist(D2_predicted, bins)

fig, ax = plt.subplots(1, 2)

ax[0, ].hist(D2_test, bins)
ax[0, ].set_title('true D2')
ax[0, ].set_xlabel('Dose (Gy)')
ax[0, ].set_ylabel('counts')
ax[1, ].hist(D2_predicted, bins)
ax[1, ].set_title('Predicted D2')
ax[1, ].set_xlabel('Dose (Gy)')
ax[1, ].set_ylabel('counts')

plt.show()


#%% 16 layer model
dropout = 0.1
kernel_initializer='glorot_uniform'

layers16_model = keras.Sequential([
    layers.Dense(20, activation='relu', input_shape=([X_train_scaled.shape[1]])),
    layers.BatchNormalization(),
    layers.Dense(40, activation='relu', kernel_initializer='glorot_uniform',bias_initializer='zeros'),
    layers.Dropout(dropout),
    layers.Dense(60, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(80, activation='relu'),
    layers.Dropout(dropout),
    layers.Dense(100, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(100, activation='relu'),
    layers.Dropout(dropout),
    layers.Dense(100, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(100, activation='relu'),
    layers.Dropout(dropout),
    layers.Dense(100, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(100, activation='relu'),
    layers.Dropout(dropout),
    layers.Dense(100, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(100, activation='relu'),
    layers.Dropout(dropout),
    layers.Dense(100, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(100, activation='relu'),
    layers.Dropout(dropout),
    layers.Dense(100, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(100, activation='relu'),
    layers.Dense(100),
    
  ])

histories['layers16'] = compile_and_fit(layers16_model, 'model/layers16')


layers16_scaled = layers16_model.predict(X_test_scaled)
layers16 = scale_output(layers16_scaled, scaler_dvh)

#DVH_functions.dvh_eval(dvh_test, layers16)

#%%
plt.figure(2)
plotter = tfdocs.plots.HistoryPlotter(metric = 'mse', smoothing_std=10)
plotter.plot(histories)

plt.ylim([0.9, 1.0])
plt.xlim([0, 25])