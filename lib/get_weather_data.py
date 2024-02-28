import glob, os
import pandas as pd
from lib.petites import dd_ashrae


def get_weather_data(path_weather):

    # Read the weather file.
    weather = pd.read_csv(path_weather + "weather.csv")
    weather.loc[:,'timestamp'] = pd.to_datetime(weather.loc[:,'timestamp'])

    # For now, we are building a model only with dry bulb temperature (degree days).
    dbt = (weather.loc[:,['timestamp', 'site_id', 'airTemperature']].groupby(by=['site_id','timestamp']).mean()).reset_index()
    dbt = dbt.pivot(index='timestamp', columns='site_id', values='airTemperature')
    for col in dbt.columns:
        if dbt.loc[:,col].isna().any():
            dbt.loc[:,col] = dbt.loc[:,col].interpolate(method='linear')

    # We are going to use Monthly HDD and CDD.
    resolution = '1ME'
    # Calculate HDD and CDD for each site.
    hdd_temp = dict()
    cdd_temp = dict()
    for col in dbt.columns:
        h, c, _ = dd_ashrae(dbt.loc[:,col], resolution=resolution)
        hdd_temp[col]=h
        cdd_temp[col]=c
    hdd_temp = pd.concat(hdd_temp)
    cdd_temp = pd.concat(cdd_temp)
    hdd_temp.name='hdd'
    cdd_temp.name='cdd'

    weather = pd.merge(hdd_temp.reset_index(), cdd_temp.reset_index(), on=['level_0', 'timestamp'], how='inner')
    weather.rename(columns={'level_0':'site_id'}, inplace=True)

    return weather