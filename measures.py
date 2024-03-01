import random

import pandas as pd
import numpy as np


def reduce_cooling_load(model:list, size:str='small'):
    # Reduce coefficient of CDD by a large or small amount. A larger reduction is more costly.

    if size == 'large':
        model[1] -= random.uniform(0.25, 0.50)*model[1]
        cost = 5000000
    elif size == 'small':
        model[1] -= random.uniform(0.05, 0.25)*model[1]
        cost = 1000000

    return model, cost


def reduce_heating_load(model:list, size:str='small'):
    # Reduce coefficient of HDD by a large or small amount. A larger reduction is more costly.

    if size == 'large':
        model[0] -= random.uniform(0.25, 0.5)*model[0]
        cost = 5000000
    elif size == 'small':
        model[0] -= random.uniform(0, 0.25)*model[0]
        cost = 1000000

    return model, cost


def reduce_base_load(model:list, size:str='small'):
    # Reduce base load (intercept) by a large or small amount. A larger reduction is more costly.

    if size == 'large':
        model[2] -= random.uniform(0.25, 0.5)*model[2]
        cost = 5000000
    elif size == 'small':
        model[2] -= random.uniform(0, 0.25)*model[2]
        cost = 1000000

    return model, cost


def pv_offset(model:list, size:str='small'):
    # Use PV to offset base load.

    if size == 'large':
        model[2] -= random.uniform(0.5, 0.75)*model[2]
        cost = 2500000
    elif size == 'small':
        model[2] -= random.uniform(0.25, 0.5)*model[2]
        cost = 500000

    return model, cost


def ppa_offset(model:list, size:str='small'):
    # Use PPA (power purchasing agreement) to offset base load.

    if size == 'large':
        model[2] -= random.uniform(0.9, 1)*model[2]
        cost = 6000000
    elif size == 'small':
        model[2] -= random.uniform(0.4, 0.5)*model[2]
        cost = 1200000

    return model, cost


