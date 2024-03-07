'''
This function simply returns the grid carbon electricity in kgCO2/kWh as a time series at annual resolution. 
The intensity is assumed to decline linearly from today to 2070.

Actual decrease from 2017 to 2021 is taken from Appendix C of https://cea.nic.in/wp-content/uploads/baseline/2023/01/Approved_report_emission__2021_22.pdf

Using a modified version of NDC target for intensity from https://climateactiontracker.org/countries/india/ by moving baseline year to 2017 (45% by 2030).

'''

import pandas as pd
import numpy as np



def grid_carbon(start_year:int=2017, end_year:int=2050, start_intensity:float=750, end_intensity:float=0.02*750):

    # projected_decrease = np.linspace(start_intensity, end_intensity, num=end_year-start_year+1)
    actual_data = pd.DataFrame(index=pd.date_range(start=f'2017-12-31', end=f'2021-12-31', freq='1YE'), columns=['kgCO2/kWh'], data = [0.75, 0.74, 0.71, 0.70, 0.72])

    intensity = pd.concat([actual_data, pd.DataFrame(index=pd.date_range(start=f'2022-12-31', end=f'{end_year}-12-31', freq='1YE'), columns=['kgCO2/kWh'])], axis=0) 

    if start_year < 2017:
        intensity = pd.concat([pd.DataFrame(index=pd.date_range(start=f'{start_year}-12-31', end=f'2016-12-31', freq='1YE'), columns=['kgCO2/kWh']), intensity], axis=0) 
    
    intensity.loc[intensity.index.year==2030,:] = 0.55*intensity.loc[intensity.index.year==2017,:].values

    # intensity.loc[intensity.index.year==2017:intensity.index.year==2021,:] = actual_data

    intensity.interpolate(method='linear', limit_area=None, limit_direction='both', inplace=True)

    return intensity