# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 17:35:00 2024

This is a pipeline for analysing some neural data with the 
new method of analysis, called dynamic RSA

You can compare the similarity of neural data (e.g. in this case: MEG data)
with other computational dynamic models.


In this case, my test subjects have seen videos of a moving dot inside MEG.
I compare the similarity (correlation) between the neural MEG data and
comptational models of the position and direction of the dot in the videos. 


@author: Marisa
"""

#import yaml
import scipy.io as sio
import numpy as np
import dRSA_coreFunction
#from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt


## Load the neural data
Yfu = sio.loadmat('example_data.mat')
Y2 = Yfu.get('Y')
Y = np.array(Y2)
#Y = Y.flatten()


# Load the model data
Modelfu = sio.loadmat('example_model.mat')
models = Modelfu['tosave']
Data_Model = [models[0, 0], models[0, 1]]  

# Specify the parameters
# Load YAML config file ?? 
# Easier: Dictionary

opt = dict()
opt = {
    'sampleDur': 0.01,  # Example: 100 Hz sampling rate, so each sample is 0.01 seconds
    'SubSampleDurSec': 5,  # Duration of each subsample in seconds
    'nIter': 10,  # Number of iterations
    'nSubSamples': 15,  # Number of subsamples
    'distanceMeasureModel': 'euclidean',  # Distance measure for model RDMs
    'distanceMeasureNeural': 'euclidean',  # Distance measure for neural RDMs
    'dRSA': {
        'corrMethod': 'corr'  # Method for correlation: 'corr' or 'PCR'
    },
    'AverageTime': 1,  # Time duration for averaging at the end in seconds
    'spacingSec': 0.1,  # Minimum distance between subsamples in seconds
    'spaceStartSec': 0.2,  # Earliest sequence start in seconds
    'modelLabel' : ['position', 'direction'] 
        }

# Add somethings that need to be calculated
opt['spacing'] = int(opt['spacingSec'] / opt['sampleDur'] ) # Minimum distance between subsamples in sample units
opt['spaceStart'] = int(opt['spaceStartSec'] / opt['sampleDur'] )  # Earliest start of subsample after NaN in sample units
opt['SubSampleDur'] = int( opt['SubSampleDurSec'] / opt['sampleDur'])
#opt['spacing'] = opt['spacingSec'] / opt['sampleDur']
opt['spacing'] = int(opt['spacingSec'] / opt['sampleDur'])  # in sec transformed into time points



olddata = Y
model_data = Data_Model

# Call the functions in my Pipeline
meandRSA = dRSA_coreFunction.dRSA_core(Y, model_data, opt)
## also, this is incredible slow. I am unsure, why, in Matlab this takes max 1-2min
# If you don't want to wait, try this:
    # SubSampleDurSec = 1
    # nIter = 3
    # nSubSamples = 5
    # that should help. 
    
    



 
## plotting
plt.figure()  # Create a new figure for each model
for iModel in range(len(model_data)):
         plt.subplot(len(model_data), 1, iModel + 1)

     
         plt.imshow(meandRSA[:, :, iModel], aspect='auto', cmap='viridis')
         plt.colorbar()  # Optional: add a colorbar
    
            # Add labels and title
         plt.xlabel('neural')
         plt.ylabel('model')
    
        # Add a title for the current model
         plt.title(f'Model {iModel+1}')  # Adjust the title as needed

     
     
     
     ### does not look at all like what I got from MAtlab...something went wrong
          