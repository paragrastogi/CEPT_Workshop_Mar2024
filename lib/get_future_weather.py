import os
import pandas as pd
import numpy as np
from lib.petites import dd_ashrae

def get_future_weather(listFutureFiles, pathSave, scenario, resolution):

    if os.path.isfile(pathSave):
        loaded = pd.read_pickle(pathSave)
        return loaded['hdd'], loaded['cdd'] 

    # TAS MIN
    filePath = [x for x in listFutureFiles if ('tasmin' in x and scenario in x)]
    tasmin = pd.read_csv(filePath[0])

    # Get index ready.
    futureIndex = pd.date_range(start = f'{tasmin.loc[:,'Year'].min()}-01-01', end = f'{tasmin.loc[:,'Year'].max()}-12-31', freq='1D')

    tasmin.set_index(futureIndex, inplace=True)
    tasmin.drop(columns=['Year', 'Day'], inplace=True)

    # TAS MAX
    filePath = [x for x in listFutureFiles if ('tasmax' in x and scenario in x)]
    tasmax = pd.read_csv(filePath[0])

    tasmax.set_index(futureIndex, inplace=True)
    tasmax.drop(columns=['Year', 'Day'], inplace=True)

    tasmid = (tasmax + tasmin)/2

    # Convert to Celsius from Kelvin.
    tasmid -= 273.15

    hdd, cdd, _ = dd_ashrae(tasmid, resolution=resolution)

    outed = {'hdd':hdd, 'cdd':cdd}

    pd.to_pickle(outed, pathSave)

    return hdd, cdd

    # # Declare a list to hold the individual dataframes for each year.
    # listDfWthr = list()

    # for file in listFutureFiles:
        
    #     wtemp, _, _ = pd.read_csv('amd', file)
    #     listDfWthr.append(wtemp)
        
    # # Create a dataframe from list but also keep the list - useful for plotting later. 
    # dfW = pd.concat(listDfWthr)


