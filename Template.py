# -*- coding: utf-8 -*-
"""
Created on Mon May 18 11:40:53 2020

@author: mauri
"""

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

#%% Functions to compile and fit the model
def get_callbacks(name):
  return [
    tfdocs.modeling.EpochDots(),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=300),
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

#%% an example model
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