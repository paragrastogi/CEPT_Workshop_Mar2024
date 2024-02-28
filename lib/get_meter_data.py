import glob, os
import pandas as pd
import numpy as np
from lib.reduce_mem_usage import reduce_mem_usage

def get_meter_data(path_meter_cleaned, path_meter_proc):

    if os.path.isfile(path_meter_proc + "allmeters_cleaned.csv"):
        data = pd.read_csv(path_meter_proc + "allmeters_cleaned.csv")
        timer = pd.to_datetime(data.loc[:,"timestamp"])
        data.drop(columns='timestamp', inplace=True)
        data.loc[:,"timestamp"] = timer
        return data

    files = glob.glob(path_meter_cleaned + "*.csv")
    dfs = [] # empty list of the dataframes to create
    for file in files: # for each file in directory
        meter_type = os.path.split(file)[-1].split("_")[0] # meter_type to rename the value feature

        # Skip water meter.
        if meter_type in ["water", "irrigation"]:
            continue

        meter = pd.read_csv(file) # load the dataset
        meter = pd.melt(meter, id_vars = "timestamp", var_name = "building_id", value_name = "meter_reading") # melt dataset
        meter["meter_type"] = str(meter_type) # adds column with the meter type
        
        # Convert timestamp to datetime.
        timer = pd.to_datetime(meter.loc[:,"timestamp"])
        # Create year and month columns to group by.
        meter.loc[:,"year"] = timer.dt.year
        meter.loc[:,"month"] = timer.dt.month
        # Drop timestamp column from meter.
        meter.drop(columns="timestamp", inplace=True)

        timer = pd.concat([timer, timer.dt.year, timer.dt.month, meter["building_id"]], axis=1)
        timer.columns = ["timestamp", "year", "month", "building_id"]

        # Save meter type column before summing since it is a text column.
        meter_type = meter.loc[:,["year", "month", "building_id", "meter_type"]]

        # Group by month and add back timestamp and meter_type.
        meter = meter.groupby(["year", "month", "building_id"]).sum()
        
        meter.loc[:,"timestamp"] = timer.groupby(["year", "month", "building_id"]).last()
        meter.loc[:,"meter_type"] = meter_type.groupby(["year", "month", "building_id"]).last()
        
        # Drop blank meter readings.
        meter.dropna(subset="meter_reading", inplace=True)

        # Flatten the index.
        meter.reset_index(inplace=True)

        # Drop time from timestamp since it is redundant. Drop month as well.
        meter.loc[:, "timestamp"] = meter.loc[:, "timestamp"].dt.date
        meter.drop(columns=["year", "month"], inplace=True)

        dfs.append(meter) # append to list
    
    data = pd.concat(dfs, axis=0, ignore_index=True) # concatenate all meter

    # Sum all meters except water and irrigation, which have been excluded already.
    data = data.loc[:,["meter_reading", "building_id","timestamp"]].groupby(["building_id", "timestamp"]).sum()
    data.reset_index(inplace=True)

    data = reduce_mem_usage(data)
    # data.loc[:,"timestamp"] = pd.to_datetime(data.loc[:,"timestamp"])

    data.to_csv(path_meter_proc + "allmeters_cleaned.csv")
    
    return data