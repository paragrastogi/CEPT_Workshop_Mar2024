# Import various standard modules.
import glob, os, copy, pickle
from datetime import datetime

# Computational modules.
import pandas as pd
import numpy as np

# Plotting modules.
import matplotlib.pyplot as plt
import seaborn as sns

# This will be used to fit and predict using a regression model.
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler 

# # Get inline graphs .
# %matplotlib inline
# # Only useful for debugging, when you 
# # need to reload external modules.
# from importlib import reload

# Enable xkcd mode if you're a geek like Parag.
# plt.xkcd()

# Import a custom read-write function for weather files.
import lib.wfileio as wf

# import a file of small helpers I've written. Call it helper.
import lib.petites as helpers

# Import an awesome colour palette. Call it colours.
import lib.default_colours as colours

# Import a function to create a portfolio of buildings to use in this exercise.
from lib.create_portfolio import create_portfolio

# Set the random seed to a specific value so the experiment is repeatable. 
# See https://en.wikipedia.org/wiki/Random_seed for more information on what this means.
# Change to your favourite number or today's date in unix timestamp format when doing the exercise yourself.
randomseed = 42 # round(datetime.timestamp(datetime.now()))

##

# Read weather data from location.

# I've used Ahmedabad as an example here - feel free to download weather data for any other city
# from http://climate.onebuilding.org/default.html if you like.

pathWthrFolder = './data/ahmedabad' 
list_wfiles = glob.glob(pathWthrFolder+'/*.epw')

# Python can interpret the Unix file separator, the 'forward-slash' (/), on all platforms. 
# That is, if you consistently use '/', the paths are automatically constructed based on the OS.
# If you want to use the Windows back-slash, make sure to precede the path string with an 'r'.

# The small program `get_weather` stores data from the incoming weather file as a dataframe. 
# It outputs three things but I'm only using the first output for now, so I've put in underscores
# to indicate that the second and third outputs should not be assigned to a variable in memory.

# Declare a list.
listDfWthr = list()

for file in list_wfiles:
    
    wtemp, _, _ = wf.get_weather('amd', file)
    listDfWthr.append(wtemp)
    
# Create a dataframe from list but also keep the list - useful for plotting later. 
dfW = pd.concat(listDfWthr)

del listDfWthr

# Calculate HDD and CDD at the given resolution (1 month). 
# Don't change this unless you rerun the training with a different resolution.
RESOLUTION = '1ME' 
hdd, cdd, _ = helpers.dd_ashrae(dfW.loc[:,'tdb'], resolution=RESOLUTION)
hdd.name = 'hdd'
cdd.name = 'cdd'

# Create a portfolio to work with for this exercise.
size = 100
btypes = ['Education', 'Office']
location = ['all']
randomness = True # This will tell the function below to inject a bit of randomness in each model so each building's response is slightly different.
portfolio = create_portfolio(size, btypes, location, path_models='./', randomness=randomness)

# Congratulations, you now have a portfolio of buildings in Ahmedabad. Each building has a unique ID, a building type (btype), and linear regression model representing its performance based on HDD and CDD.

##
# Maybe train different models using different train-test splits for each building in the portfolio instead of always 2016 and 2017?
##

# Calculate the present performance of the portfolio.
X = pd.merge(hdd, cdd, how='inner', left_index=True, right_index=True)
X.fillna(0, inplace=True)

# We need to scale the inputs to have zero mean and unit variance (1). 
# This is because the model was fit with these transformed features. 
# Transforming features or inputs in this way makes it easier to use variables with 
# potentially very different magnitudes together in one equation and keep them comparable.
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)
X_scaled = pd.DataFrame(X_scaled, columns = X.columns, index=X.index)

perf = portfolio.loc[:,'model'].apply(lambda x: ([item[0] for item in x.predict(X_scaled)]))
perf.name = 'performance'
performanceHist = pd.DataFrame(perf.tolist(), index=portfolio.index, columns=X.index)

total_performance = performanceHist.sum(axis=1)

# Calculate the future performance of the portfolio without any changes to buildings or composition of portfolio.


print(portfolio)