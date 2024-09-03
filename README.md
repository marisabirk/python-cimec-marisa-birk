# Exemplary Dynamic Representation Similarity Analysis (dRSA) Pipeline

This repository provides an exemplary pipeline for dynamic Representation Similarity Analysis (dRSA) of some data. 
Dynamic RSA presents a new dynamic extension to representational similarity analysis (RSA) which allows investigating when naturalistic dynamic stimuli are represented in the brain by comparing the similarity between dynamic computational model features and dynamic neural data.

In 'data' I provide neural MEG data from a test subject measured in June 2024. In this experiment, the participant watched videos of a dot moving across the screen.
I also provide data from two computational model features (position, direction) and the neural data.  


## Overview

- **Data_Analysis.py**: The main pipeline script for the analysis that loads the example data, provides the configuration file and calls other functions and plots the results.
- **dRSA_coreFunction.py**: The core function of the analysis, capable of comparing any dynamic neural and computational data.
- **dRSA_subsampling2.py**: A script for creating subsamples to generalize the data.

## Installing

To use this pipeline, follow these steps:

1. Download all files from this repository.
2. Ensure all files are placed within the same folder on your local machine.

## Usage

1. Ensure that the working directory is in the folder where you placed all your files.
2. Run `Data_Analysis.py` to start the main analysis.


## Example Data

The repository contains:
- Example neural data from a test subject.
- Example data from two models (position, direction) used in the dRSA.

You can replace this example data with your own data to perform similar analyses.
