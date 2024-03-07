# Import various standard modules.
import glob, os, copy, pickle
from datetime import datetime

# Computational modules.
import pandas as pd
import numpy as np

# Plotting modules.
import matplotlib.pyplot as plt
import seaborn as sns
# Change backend for Mac
if os.name == 'posix':
    import matplotlib
    matplotlib.use('TKAgg')

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

from lib.get_future_weather import future_weather
from lib.get_financials import income_value, electricity_cost
from lib.get_carbon import grid_carbon
from lib.get_measures import apply_measures

# Set the random seed to a specific value so the experiment is repeatable. 
# See https://en.wikipedia.org/wiki/Random_seed for more information on what this means.
# Change to your favourite number or today's date in unix timestamp format when doing the exercise yourself.
randomseed = 42 # round(datetime.timestamp(datetime.now()))

##

# Read weather data from location.
station = 'Glasgow'
epwFolder = 'GBR_SCT_Glasgow.Intl.AP.031400'

TARGET_INTENSITY_DECREASE = 0.98 # 98% decrease in energy intensity for buildings.
TARGET_YEAR = 2050 # Year by which target intensity decrease must be achieved.

PATH_MEASURES_FILE = "/Users/prastogi/Library/CloudStorage/OneDrive-Personal/CEPT/Workshop-2024/DecarbPlan.xlsx"

# I've used Glasgow as an example here and we will use Ahmedabad for the exercise. 
# However, feel free to download weather data for any other city
# from http://climate.onebuilding.org/default.html if you like.

pathWthrFolder = '/Users/prastogi/Library/CloudStorage/OneDrive-Personal/CEPT/Workshop-2024/Data/WeatherData/' 

listWfiles = glob.glob(f'{pathWthrFolder}/{epwFolder}/*.epw')
# Python can interpret the Unix file separator, the 'forward-slash' (/), on all platforms. 
# That is, if you consistently use '/', the paths are automatically constructed based on the OS.
# If you want to use the Windows back-slash, make sure to precede the path string with an 'r'.

# The small program `get_weather` stores data from the incoming weather file as a dataframe. 
# It outputs three things but I'm only using the first output for now, so I've put in underscores
# to indicate that the second and third outputs should not be assigned to a variable in memory.

# Declare a list to hold the individual dataframes for each year.
listDfWthr = list()

for file in listWfiles:
    
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

# Plot historical CDD. You can also plot HDD but they are so few and far between in Ahmedabad it's pointless. To keep the graph relatively clutter-free, take the sum of performance over each year and plot that instead of monthly values.

ploty = cdd.resample('1YE').sum()
fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=True, sharey=True, figsize=[12,8])
ax = axes #.flatten()
fig.tight_layout(pad=3)
plt.bar(x=ploty.index, height=ploty, width=300)
# plt.show()

# Create a portfolio to work with for this exercise.
PORTFOLIO_SIZE = 100
TYPES_INCL = ['Education', 'Office']
LOCATIONS = ['all']
RANDOMNESS = True # This will tell the function below to inject a bit of randomness in each model so each building's response is slightly different.
portfolio = create_portfolio(PORTFOLIO_SIZE, TYPES_INCL, LOCATIONS, path_models='./', randomness=RANDOMNESS)

# Congratulations, you now have a portfolio of buildings in Ahmedabad. Each building has a unique ID, a building type (btype), and linear regression model representing its performance based on HDD and CDD.

##
# Maybe train different models using different train-test splits for each building in the portfolio instead of always 2016 and 2017?
##

# Calculate the present performance of the portfolio.

# Merge HDD and CDD into a single dataframe.
X = pd.merge(hdd, cdd, how='inner', left_index=True, right_index=True)
X.dropna(how='any', inplace=True)

# We need to scale the inputs to have zero mean and unit variance (1). 
# This is because the model was fit with these transformed features. 
# Transforming features or inputs in this way makes it easier to use variables with 
# potentially very different magnitudes together in one equation and keep them comparable.
# Standard scaler takes an element x_i and converts it to z_i = (x_i - \mu)/\sigma .

scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)
X_scaled = pd.DataFrame(X_scaled, columns = X.columns, index=X.index)

performanceHistorical = portfolio.loc[:,'model'].apply(lambda x: x[0]*X_scaled.iloc[:,0] + x[1]*X_scaled.iloc[:,1] + x[2])

performancePortfolioHistorical = performanceHistorical.sum(axis=0)
performancePortfolioHistorical.name = 'performance'

## 
# Plot historical performance. To keep the graph relatively clutter-free, take the sum of performance over each year and plot that instead of monthly values.

ploty = performancePortfolioHistorical.resample('1YE').sum()
fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=True, sharey=True, figsize=[12,8])
ax = axes #.flatten()
fig.tight_layout(pad=3)
plt.bar(x=ploty.index, height=ploty, width=300)
# plt.show()

# Calculate the future performance of the portfolio without any changes to buildings or composition of portfolio.
SCENARIO = 'ssp585'
listFutureFiles = glob.glob(f'{pathWthrFolder}/{station}_CMIP6/*.csv')
pathSave = f'{pathWthrFolder}/future_dd.pickle'

hddFuture, cddFuture = future_weather(listFutureFiles, pathSave, scenario=SCENARIO, resolution=RESOLUTION)

ploty1 = cdd.resample('1YE').sum()
ploty2 = cddFuture.resample('1YE').sum().rolling('1200D').mean()
# ploty2 = cddFuture.rolling(window='360D').sum()
fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=True, sharey=True, figsize=[12,8])
ax = axes #.flatten()
fig.tight_layout(pad=3)
plt.plot(ploty2.index, ploty2, color=colours.orange)
plt.plot(ploty1.index, ploty1, color=colours.blue)

listpf = list()
for ccmodel in hddFuture.columns:
    X = pd.merge(hddFuture.loc[:,ccmodel], cddFuture.loc[:,ccmodel], how='inner', left_index=True, right_index=True)
    X.dropna(how='any', inplace=True)

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns = X.columns, index=X.index)

    performanceFuture = portfolio.loc[:,'model'].apply(lambda x: x[0]*X_scaled.iloc[:,0] + x[1]*X_scaled.iloc[:,1] + x[2])

    perfPF = performanceFuture.sum(axis=0)
    perfPF.name = ccmodel

    listpf.append(perfPF)

performancePortfolioFuture = pd.concat(listpf, axis=1)


ploty1 = performancePortfolioHistorical.resample('1YE').sum()
ploty2 = performancePortfolioFuture.resample('1YE').sum().rolling('1200D').mean()
# ploty2 = cddFuture.rolling(window='360D').sum()
fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=True, sharey=True, figsize=[12,8])
ax = axes #.flatten()
fig.tight_layout(pad=3)
plt.plot(ploty2.index, ploty2, color=colours.orange)
plt.plot(ploty1.index, ploty1, color=colours.blue)

# Let's simplify the future performance projection to end in 2050 and clip the years that have already passed so we can join it to the measured/historical performance.
performancePortfolioFutureMean = performancePortfolioFuture.mean(axis=1)
performancePortfolioFutureMean.name='mean'
performancePortfolioFutureMean = performancePortfolioFutureMean.loc[(performancePortfolioFutureMean.index.year>performancePortfolioHistorical.index.year.max()) | (performancePortfolioFutureMean.index.year<TARGET_YEAR)]

performancePortfolio = pd.DataFrame(pd.concat([performancePortfolioHistorical, performancePortfolioFutureMean]))
performancePortfolio.sort_index(inplace=True)
performancePortfolio.columns = ['consumption_kWh']

# Get grid carbon emissions factors. These vary over time and are projected into the future.
intensityCurve = grid_carbon(start_year=performancePortfolio.index.year.min(), end_year=TARGET_YEAR)

# Get the constant electricity cost, income per building, and value of each building.
unitCost = electricity_cost()
incomePerBuilding, valueperBuilding = income_value()

performancePortfolio = pd.concat([performancePortfolio, pd.DataFrame(columns=['size'],index=performancePortfolio.index,data=PORTFOLIO_SIZE)], axis=1)
performancePortfolio.loc[:,'income'] = performancePortfolio.loc[:,'size'] * incomePerBuilding
performancePortfolio.loc[:,'value'] = performancePortfolio.loc[:,'size'] * valueperBuilding
performancePortfolio.loc[:,'runningCosts'] = performancePortfolio.loc[:,'consumption_kWh']*unitCost

# Apply the measures you've specified in your Excel file.
performancePortfolio = apply_measures(performancePortfolio, PATH_MEASURES_FILE)

print(portfolio)