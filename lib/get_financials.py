'''
This script contains functions to get the cost of electricity, 
income from a building, and the value of each building. For now, 
these are simple constants but can be upgraded to whatever 
function is required.
'''


def income_value():

    incomePerBuilding = 1000000 # INR 10 lakh 
    valuePerBuilding = 15000000 # INR 1.5 Cr

    return incomePerBuilding, valuePerBuilding


def electricity_cost():
    # INR 6.15/kWh in FY20 to INR 5.25/kWh and INR 5.4/kWh by 2050

    unitCost = 6.15 # INR 6.15/kWh

    return unitCost