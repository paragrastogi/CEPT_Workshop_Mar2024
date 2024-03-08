import random
import copy
import pandas as pd
import numpy as np
from lib.measures import apply_CDD_measure, apply_HDD_measure, apply_BASE_LOAD_measure, apply_PV_measure, apply_PPA_measure
from lib.get_performance import calc_performance


MEASURE_TYPES = ['CDD', 'HDD', 'BASE_LOAD', 'PV', 'PPA']


def apply_measures(portfolio, X, pathMeasuresFile):

    portfolio_future = copy.copy(portfolio)

    measuresPlan = pd.read_excel(pathMeasuresFile, sheet_name='PLAN - EDIT THIS', usecols='A:R', nrows=27, index_col=0)

    measuresPlan.index = pd.to_datetime(measuresPlan.index, format='%Y')

    costsPortfolio = list()
    perfPortfolio = list()

    for row in measuresPlan.iterrows():

        costs_temp = list()

        for measure in MEASURE_TYPES:
            
            buildingsAffectedNumber = row[1].loc[f'{measure} Number']

            buildingsAffectedIndex = random.sample(list(portfolio_future.index), buildingsAffectedNumber)

            temp_out = portfolio_future.loc[buildingsAffectedIndex,"model"].apply(lambda x: eval(f'apply_{measure}_measure(x, "{pathMeasuresFile}")'))

            costs_temp.append(temp_out.apply(lambda x: x[1]).sum())
            portfolio_future.loc[buildingsAffectedIndex,"model"] = temp_out.apply(lambda x: x[0])

        perf = portfolio_future.loc[:,'model'].apply(lambda m: calc_performance(m, X.loc[X.index.year==row[0].year,:]))

        perfPortfolio.append(perf)
        costsPortfolio.append(sum(costs_temp))

    perfPortfolio = pd.concat(perfPortfolio, axis=1)
    costsPortfolio = (pd.DataFrame([costsPortfolio, np.cumsum(costsPortfolio)], index=['AnnualCosts', 'CumulCosts'], columns=measuresPlan.index)).T

    return perfPortfolio, costsPortfolio