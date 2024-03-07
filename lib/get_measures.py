import pandas as pd

MEASURE_TYPES = ['CDD', 'HDD', 'BASE_LOAD', 'PV', 'PPA']

def apply_measures(performancePortfolio, pathMeasuresFile):

    measuresPlan = pd.read_excel(pathMeasuresFile, sheet_name='PLAN - EDIT THIS', usecols='A:R', nrows=27, index_col=0)

    measuresPlan.index = pd.to_datetime(measuresPlan.index, format='%Y')

    for row in measuresPlan.iterrows():
        for measure in MEASURE_TYPES:
            
            buildingsAffected = row.loc[measure]

            

    return performancePortfolio