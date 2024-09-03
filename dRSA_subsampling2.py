# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 23:32:03 2024

@author: Marisa
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 17:32:44 2024

To generalize data, we apply a sort of bootstrapping process. 
X Samples are taken from our data, and the similarity between
computational and neural models is calculated. 

We repeat this process across iIter Iterations and then average in the end

@author: Marisa
"""

#dRSA_subsampling2


import numpy as np
import pandas as pd


def dRSA_createSubsamples(maskSubsampling, opt):
    """ The Mask is a vector telling us from where we are allowed to 
    take Subsamples. e.g. we don't want to take one, where one video
    ends and a new one starts   
    
    Input arguments:
    maskSubsampling = 1xN logical vector with 1 = masked out time points (final subsamples should not overlap with TPs = 1)
    opt = configurational parameters
     
    nSubsamples = (how many subsamples should be selected from Event)
    SubSampleDur = how many time points per Subsample; must be smaller than length of mask
    nIter = number of Iterations
    spacing = minimum distance between Subsamples
    
    Output Arguments:
    SSIndices = array with dim a) num of subsamples, b. num of time points per Sample, c). num of iterations
    SSIndicesPlot = for plotting 
    """
    # Ensure the mask has enough space for subsamples
    #maskSubsampling = maskSubsampling.astype(bool)
    # Initialize the empty arrays

    
    SSIndices = np.empty((opt['nSubSamples'], opt['SubSampleDur'], opt['nIter']), dtype='object')
    SSIndicesPlot = np.zeros((opt['nIter'], len(maskSubsampling)), dtype=int)

    for iIter in range(opt['nIter']):
        TransformedMask = maskSubsampling.copy()
        constraints = 0

        while constraints == 0:
            SSstarts = []
            sample = 1  # I know Python starts counting with 0, but this is more logical for me
            totalrestart = 0  # to avoid an endless loop
            newSampleRestart = 0  # Restart Choosing Subsamples
            constraint2 = 0  # my second constraint for the while loop

            while sample <= opt['nSubSamples']:
                if totalrestart > 20:  # to avoid endless loop
                    print(f'Iteration {iIter}: No solution found after 20 total restarts')
                    break

                elif newSampleRestart > 50:
                    sample = 1
                    SSstarts = []
                    TransformedMask = maskSubsampling.copy()
                    newSampleRestart = 0
                    totalrestart += 1
                    constraint2 = 0
                    print(f'Iteration {iIter}: Restarting sample placement after 50 attempts')

                else:

                    try:
                        SSstart = np.random.choice(np.where(TransformedMask == 0)[0])  # give me the index
                        constraint2 = 1
                    except:
                        print(f'Iteration {iIter}: No space left to place Subsamples')
                        sample = 1
                        SSstarts = []
                        TransformedMask = maskSubsampling.copy()
                        constraint2 = 0
                        totalrestart += 1

                    if constraint2 == 1:
                        SSend = SSstart + opt['SubSampleDur'] + opt['spacing']

                        if SSend < len(TransformedMask) and np.all(TransformedMask[SSstart:SSend] == 0):
                            TransformedMask[SSstart:SSend] = 1
                            SSstarts.append(SSstart)
                            newSampleRestart = 0
                            sample += 1
                        else:
                            newSampleRestart += 1

            SSstarts = np.array(sorted(SSstarts))
            SSends = SSstarts + opt['SubSampleDur']
            gaps = SSstarts[1:] - SSends[:-1]
            idx = np.array([np.arange(start, start + opt['SubSampleDur']) for start in SSstarts])

            if len(gaps) > 0 and np.min(gaps) >= opt['spacing'] and np.all(np.isin(idx, np.where(maskSubsampling == 0)[0])) and np.max(idx) < len(maskSubsampling):
                SSIndices[:, :, iIter] = idx
                SSIndicesPlot[iIter, idx.flatten()] = 2
                constraints = 1
                print(f'Iteration {iIter}: Solution found')

    return SSIndices, SSIndicesPlot







