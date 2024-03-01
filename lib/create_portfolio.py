'''
Compose a portfolio with a fixed size by sampling different building types and/or regions.
'''


import glob, os
import copy
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

def create_portfolio(size:int=100, btypes:list='all', location:list='all', path_models:str='./', randomness:bool=True):

    with open(path_models + 'models.pickle', 'rb') as file:
        models = pickle.load(file)
    
    if btypes[0] == 'all':
        models_selected = models
    else:
        models_selected = models.loc[models.loc[:,'btype'].isin(btypes),:]

    portfolio = models.loc[random.choices(models_selected.index,k=size),:]

    if randomness:
        portfolio.loc[:,'model'] = portfolio.loc[:,'model'].apply(lambda x: inject_randomness(x))

    building_id = pd.RangeIndex(1, portfolio.shape[0]+1)
    portfolio.set_index(building_id, inplace=True)

    return portfolio



def inject_randomness(mdl):
    # inject a bit of randomness into the learned coefficients.
    
    # mdl.coef_[0] = mdl.coef_[0] + randomcoeffs[:2]
    # mdl.intercept_ = mdl.intercept_ + randomcoeffs[-1]

    print(mdl)

    jitters = [random.normalvariate(0,1) for x in mdl]

    mdl = [x+y for x,y in zip(mdl, jitters)]

    return mdl