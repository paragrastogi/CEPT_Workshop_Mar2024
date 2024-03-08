import random

import pandas as pd
import numpy as np

# filePath = '/Users/prastogi/Library/CloudStorage/OneDrive-Personal/CEPT/Workshop-2024/DecarbPlan.xlsx'


def apply_CDD_measure(model:list, pathMeasuresFile:str, size:str='small'):
    # Reduce coefficient of CDD by a large or small amount. A larger reduction is more costly.

    read_costs = pd.read_excel(pathMeasuresFile, sheet_name='KEYS - DO NOT EDIT', usecols='B:G', nrows=3, index_col=0)

    feature = 'CDD'

    if size == 'large':
        model[1] -= random.uniform(0.25, 0.50)*model[1]
        cost = read_costs.loc[size, feature]
    elif size == 'small':
        model[1] -= random.uniform(0.05, 0.25)*model[1]
        cost = read_costs.loc[size, feature]

    return model, cost


def apply_HDD_measure(model:list, pathMeasuresFile:str, size:str='small'):
    # Reduce coefficient of HDD by a large or small amount. A larger reduction is more costly.

    read_costs = pd.read_excel(pathMeasuresFile, sheet_name='KEYS - DO NOT EDIT', usecols='B:G', nrows=3, index_col=0)

    feature = 'HDD'

    if size == 'large':
        model[0] -= random.uniform(0.25, 0.5)*model[0]
        cost = read_costs.loc[size, feature]
    elif size == 'small':
        model[0] -= random.uniform(0, 0.25)*model[0]
        cost = read_costs.loc[size, feature]

    return model, cost


def apply_BASE_LOAD_measure(model:list, pathMeasuresFile:str, size:str='small'):
    # Reduce base load (intercept) by a large or small amount. A larger reduction is more costly.

    read_costs = pd.read_excel(pathMeasuresFile, sheet_name='KEYS - DO NOT EDIT', usecols='B:G', nrows=3, index_col=0)

    feature = 'BASE_LOAD'

    if size == 'large':
        model[2] -= random.uniform(0.25, 0.5)*model[2]
        cost = read_costs.loc[size, feature]
    elif size == 'small':
        model[2] -= random.uniform(0, 0.25)*model[2]
        cost = read_costs.loc[size, feature]

    return model, cost


def apply_PV_measure(model:list, pathMeasuresFile:str, size:str='small'):
    # Use PV to offset base load.

    read_costs = pd.read_excel(pathMeasuresFile, sheet_name='KEYS - DO NOT EDIT', usecols='B:G', nrows=3, index_col=0)

    feature = 'PV'

    if size == 'large':
        model[2] -= random.uniform(0.5, 0.75)*model[2]
        cost = read_costs.loc[size, feature]
    elif size == 'small':
        model[2] -= random.uniform(0.25, 0.5)*model[2]
        cost = read_costs.loc[size, feature]

    return model, cost


def apply_PPA_measure(model:list, pathMeasuresFile:str, size:str='small'):
    # Use PPA (power purchasing agreement) to offset base load.

    read_costs = pd.read_excel(pathMeasuresFile, sheet_name='KEYS - DO NOT EDIT', usecols='B:G', nrows=3, index_col=0)

    feature = 'PPA'

    if size == 'large':
        model[2] -= random.uniform(0.9, 1)*model[2]
        cost = read_costs.loc[size, feature]
    elif size == 'small':
        model[2] -= random.uniform(0.4, 0.5)*model[2]
        cost = read_costs.loc[size, feature]

    return model, cost


