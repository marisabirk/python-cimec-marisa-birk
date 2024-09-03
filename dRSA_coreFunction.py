# -*- coding: utf-8 -*-
"""
 Core Function of  dynamic Representation Analysis
 
 
 Input: 
     a) Y:  independent variable, e.g. neural or behavioral data: 4-D num array with dimensions ERFT = (E)vents x (R)epetitions x (F)eatures x (T)ime points
       this can be:
          - 1x1x1xN (a single event with a single feature, e.g. pupil dilation during watching a movie, or a single neuron during a single free moving event)
          - 1x1xMxN (same as above but with >1 features, e.g. xy eye positions, or several neurons)
          - 1xLxMxN (same as above but with >1 repetitions of the same event. Data will be averaged to 1x1xMxN)
          - KxLxMxN (>1 event, e.g. several trials, from which the subsampling will be done. Data will be concatenated to 1x1xMxN)
          - mix of above, e.g. Kx1xMxN (several events without repetition)
     
    b) models: (1xN cell array, with 3-D num arrays with dimensions EFT in each cell that match Y in number of Events and Time points)
      Y and models may contain NaN.
    
    opt: Dictionary to config
    
    
Output:
    dRSA = 3-D NxNxM num array, where N = opt.SubSampleDur and M = num of tested models
 
"""

import numpy as np
import pandas as pd
import scipy.spatial.distance as spd
import matplotlib.pyplot as plt
import dRSA_subsampling2
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr

def dRSA_core(Y, model_data, opt):
    origDims = Y.shape
    nEvents, nReps, nFeatures, nTPevents = origDims
    print(f'data: {nEvents} event(s), {nReps} repetition(s), {nFeatures} feature(s), {nTPevents} time points')
    
     
    
## Average across repetitions
    if nReps > 1:
        print('average across repetitions')
        Y = np.nanmean(Y, axis=1) # hopefully correct axis
        
    Y = np.squeeze(Y)  # Remove simpleton Dimension of Repetitions
    if nFeatures == 1:
        Y = Y.T # needs to be transposed if only one feature so that Y is 1xN
        #
        
    # Concenating
    if nEvents > 1:
        print('concatenate events in data')
        #Y = Y.reshape(nFeatures, nEvents * nTPevents)      
        # Not sure if transposing necessary, in Matlab I used shiftdim
        Y = Y.transpose(2, 0, 1).reshape(nTPevents * nEvents, nFeatures).T 
        # we put them together and Transpose
        # We have features x time points
        
        print('concatenate events in models')
        for i in range(len(model_data)):                       
            model_data_temp = model_data[i].copy()
            #print(f"Original shape of model_data[{i}]:", model_data_temp.shape)
            nModelFeatures = model_data_temp.shape[1]  # how many features
            
            model_data_temp = model_data_temp.transpose(2, 0, 1).reshape(nTPevents* nEvents, nModelFeatures).T
            
            model_data[i] = model_data_temp.copy() # better to copy I guess
            
    
    
    nTP = Y.shape[1]        # Update the Time Points
    print(f'total number of time Points: {nTP}')
      
      # Create the Mask for Subsamples
    mask = np.zeros((2 + len(model_data), nTP), dtype=int)


    if nEvents > 1:
          eventStartVec = np.zeros(nTP)
          eventStartVec[np.arange(0, nTP, nTPevents)] = 1
          mask[0,:] = eventStartVec  # so that the start of each of the events is 1
          # in mask
    else:
          mask[0,0] = 1  # First Time Point is start of event
      
        
      # Add masking of TPs after event start points so that we don't get 
      # a subsample from there
    StartEvent =  np.where(mask == 1) [1] # Tuple, 1 is column
    # So in theory I should get index where mask is 1
    for idx in StartEvent:
          StartIndex = int(idx)
          EndIndex = int(idx + opt['spaceStart'])
          mask[0,idx:EndIndex] = 1
            
    maskLabels = list()
    maskLabels.append('event start')
      
      
      # NaNs in Y Data  # there are none
    mask [1,:] = (np.any(np.isnan(Y), axis=0))
    maskLabels.append('NaNs in Data')

      # NaNs in Models
      # there also none, but there could be
    for i in range(len((model_data))):
          mask[2+i,:] = (np.any(np.isnan(model_data[i]), axis=0))
          maskLabels.append ( f'NaNs in model {opt["modelLabel"][i]}')
          #maskLabel.append(opt['modelLabel'][i])
             #f'NaNs in model ({models["labels"][model_idx]})'
             
     # Turn into one dimension mask
    maskSubsampling = np.any(mask, axis = 0)
    #the mask shows us where not to place subsamples
    
    TotalDurSeqS = (opt['SubSampleDur'] * opt['nSubSamples'] +
                opt['spacing'] * (opt['nSubSamples'] - 1) +
                np.sum(maskSubsampling == 1))


    #if TotalDurSeqS > nTP:
         #print('too many subsamples/ subsamples too long')
        # return
     
    
   # Call the Subsampling Function to create Subsamples
    
    SSIndices, SSIndicesPlot = dRSA_subsampling2.dRSA_createSubsamples(maskSubsampling, opt)

     # Plotting my Subsamples
    
    #### NOT WORKING AS INTENDED BUT ALSO NOT IMPORTANT
    # The second Subplot was supposed to show my masks
    # But although I have "1" in the beginning it does not show??
    
    
    plt.figure(figsize=(10, 15))

    # # Subplot 1: Data
    plt.subplot(3, 1, 1)
    plt.imshow(Y, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.xlabel('time points')
    plt.ylabel('features')
    plt.title('data')

    # # Subplot 2: Mask
    
    plt.subplot(3, 1, 2)

    plt.imshow(mask, aspect='auto', cmap='viridis')
    plt.colorbar() 

    plt.xlabel('time points')
    plt.yticks(ticks=np.arange(len(maskLabels)), labels=maskLabels)
    plt.title('mask')

    # # Plot 3: subsampling iterations
    plt.subplot(3, 1, 3)
    plt.imshow(SSIndicesPlot, aspect='auto', cmap='viridis')
    plt.xlabel('time points')
    plt.ylabel('iterations')
    plt.title(f"{opt['nSubSamples']} subsamples, dur: {opt['SubSampleDur']} time points")

    plt.tight_layout()
    plt.show()
          
     
     
     # Correlating my neural Data and my Model Data
     # Create empty arrays:
     
     # Model Representational Dissimilalrity Matrix
     # List with empty arrays 
     #mRDMs = [np.zeros((nTP, opt['SubSampleDur'])) for _ in range(len(model_data))] 
   
    dRSA = np.zeros((opt['nIter'], opt['SubSampleDur'], opt['SubSampleDur'], len(model_data)))
     
     # dRSA 
     
     
     
     # First Loop is the Subsampling Loop to Generalize across Iterations
    for i in range (opt['nIter']):
         print(f'\n run dRSA: {i+1:04d} ')  # to know where we are
         mRDMs = []
      
         nRDMs = list()    # Neural RDM
         
         # Next we correlate the models. For each Time Point within our Data, 
         # we choose the euclidean distance between each subsamples
                  # of my RDM
         for iModel in range(len(model_data)):  # across Models
         
           mRDMs_for_model = []                     
           # First we correlate the euclidean Distance ebtween Models for each
           # Time Point between Subsamples and 
           # and we use the squareform to transform the lower square
           
           for iT in range(opt['SubSampleDur']): #iT = i from Time
                # At the moment we have 2 features for 12000 time points
                
                # We choose the Subsamples we created.
                # SSIndices  =  Subsamples x  time Points x  Iterations
                Subsample = (SSIndices[:, iT, i])  #  for that Time and Iteration
                Subsample = Subsample.astype(int)  # otherwise I get errors
                currentModel =(model_data[iModel] [:, Subsample] ).T
                
                # Transpose for pairwise distance function 
                distances  = ( pdist(currentModel, metric='euclidean'))
                mRDMs_for_model.append(distances)
                
                #mRDMs_temp = np.array(mRDMs_for_model).T
     
       
           mRDMs.append(mRDMs_for_model)
         mRDMs = np.array(mRDMs)
           
           # This should have the dimensions:
               # 2 Models x 500 time points x 105 features (if I use 15 subsamples and 5s)
                # We had a 15 x 15 comparisons. This matrix is symmetrical
                # because comparing Subsample 1 with Subsample 2 is the same as 
                # Subsample 2 with Subsample 1, 
                # And the diagonal is just comparing with itsself, so useless 
                # if you take the lower square of
                # the matrix without the diagonal you get
                # 14 + 13 +12 +11 .... + 1 = 105
                # Basically, squareform
                
                
          
         # Now for the neural model
         # Again first we go throug the Time Points
         for iT in range(opt['SubSampleDur']): #iT = i from Time
             Subsample = SSIndices[:, iT, i]  # Select the subsample indices
             Subsample = Subsample.astype(int)  # integers
             currentNeural = Y[:, Subsample].T
             
             # We take the correlation, this time not euclidean distance
             corr = pdist(currentNeural, 'correlation') 
             nRDMs.append(corr)
         nRDMs = np.array(nRDMs)
         
         # 500 time points x 105 features
         
         
         
         # Now we correlate the computational models with the neural data
        
        # For each of the 500 time points, I want to correlate the 105 neural x 105 model features
        # So I get a 500 x 500 matrix 
         # In the end: dRSA has dimensions 100 iterations, 500 neural time points, 500 model time points, 2 Models
         
       # Theres no simple function like in Matlab...
       # For loops
   
         for iModel in range(len(model_data)): 
             
             #correlations = np.zeros(( opt['SubSampleDur'],  opt['SubSampleDur']))
             
             for row_m in range(opt['SubSampleDur']):
                 for row_n in range(opt['SubSampleDur']):
                     
                     # the row of the models
                     modelRDM_temp = mRDMs[iModel, row_m,:]
                     
                     # the other rows of the neural model
                     neuralRDM_temp = nRDMs[row_n,:]
                     
                     # Peearson Correlation for the columns
                     corr, x = pearsonr(modelRDM_temp , neuralRDM_temp)  # we correlate the 105 features
                     # corr should be our correlation, no idea what x is but we need it for this to work
                     dRSA[i, row_m, row_n, iModel] = corr
                     
                     ## this takes ages but no idea how else I could do it.
                     # Matlab has a function (pdist) and does it in like in 1s

                 
       
     # now we have to take the mean across subsamples
    meandRSA = np.mean(dRSA, axis=0)
    return meandRSA 
     
     
    
      

        
        



