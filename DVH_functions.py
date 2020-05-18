# -*- coding: utf-8 -*-
"""
Created on Wed May 13 13:34:52 2020

@author: mauri

This file contains functions for the Bachelor Thesis on Machine Learning in interplay breathing step 1.
"""

# function to retrieve output Y from scaled output Y
""""Y_scaled is the scaled data from which the unscaled data is to be retrieved
    scaler_Y is the scaler that is used to retrieve the data from Y_scaled"""
   
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt

#%%
def scale_output(Y_scaled, scaler_Y):
    Y = Y_scaled*scaler_Y.scale_ + scaler_Y.mean_
    return Y

#%% function dvh evaluation

def dvh_eval(test, prediction):
    ones = np.ones((test.shape[0], test.shape[1]))
    error_rel = ones-prediction/test
    error_rel_mean = np.mean((error_rel), axis = 1)
    error_rel_min = np.amin(error_rel, axis = 1)
    error_rel_max = np.amax(error_rel, axis = 1)
    
    bins = 160
      
    figure, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
    
    ax1 = plt.subplot(311)
    plt.hist(error_rel_min, bins, range=[-0.08, 0.08])
    #plt.setp(ax1.get_xticklabels())
    plt.title(('minimal relative error'))
    
    ax2 = plt.subplot(312, sharex=ax1)
    plt.hist(error_rel_mean, bins, label = 'mean relative error', range=[-0.08, 0.08] )
    #plt.setp(ax2.get_xticklabels())
    plt.title(('mean relative error'))
    plt.ylabel('counts')
    
    ax3 = plt.subplot(313, sharex=ax1)
    plt.hist(error_rel_max, bins, range=[-0.08, 0.08])
    plt.title(('maximal relative error'))
    plt.xlabel("relative error")
    plt.subplots_adjust(hspace=0.7)
    
    
    error_evals = np.array([0.015, 0.01, 0.005, 0.0025])
    errors = np.zeros([len(error_evals)])
    for i in range(len(error_evals)):
        errors[i] = np.count_nonzero(np.abs(error_rel_mean) > (0+error_evals[i]))
        perc = errors/len(error_rel_mean)*100
    
    table = tabulate([[">{}".format(error_evals[0]), errors[0], perc[0]],
                      [">{}".format(error_evals[1]), errors[1], perc[1]],
                      [">{}".format(error_evals[2]), errors[2], perc[2]],
                      [">{}".format(error_evals[3]), errors[3], perc[3]]],
                         headers=['mean relative error', 'counts', 'percentage'])
    
    print(table)
    return
    
    
    
#%% define function to plot a dvh test and a dvh prediction
""""test is the test data
    prediction is the prediction data from the model""" 
def plot_dvh(test, prediction):
    test_number = int(np.random.rand(1,1)*test.shape[0])
    perc = np.linspace(100,1,100)
    perc = perc.reshape(100,1)
    plt.xlabel('Dose(Gy)')
    plt.ylabel('Percentage (%)')
    plt.title('Computed DVH and Predicted DVH')
    plt.plot(test[test_number], perc)
    plt.plot(prediction[test_number], perc)
    plt.legend(('computed DVH', 'Predicted DVH'))
    plt.show()
#%%
def get_callbacks(name):
  return [
    tfdocs.modeling.EpochDots(),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=75),
    #keras.callbacks.TensorBoard(logdir/name),
  ]
#%%
def compile_and_fit(model, name, max_epochs=1000):
  optimizer = tf.keras.optimizers.RMSprop(0.1)
  model.compile(loss='mse',
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