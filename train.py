import glob, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lib.get_metadata import get_metadata
from lib.get_meter_data import get_meter_data
from lib.get_weather_data import get_weather_data

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, Normalizer, StandardScaler
from sklearn.metrics import root_mean_squared_error, median_absolute_error

path_metadata = "../buds/data/metadata/"
path_weather = "../buds/data/weather/"
path_meter_raw = "../buds/data/meters/raw/"
path_meter_cleaned = "../buds/data/meters/cleaned/"
path_meter_proc = "../buds/data/meters/processed/" 


metadata = get_metadata(path_metadata)
weather = get_weather_data(path_weather)
data = get_meter_data(path_meter_cleaned, path_meter_proc)

data = pd.merge(data, metadata, on='building_id', how='left', validate='many_to_one')
data = pd.merge(data, weather, how='left', on=["timestamp", "site_id"], validate='many_to_one')

# This is likely due to a reset_index somewhere.
if 'Unnamed: 0' in data.columns:
    data.drop(columns='Unnamed: 0', inplace=True)

data.dropna(how='any', inplace=True)

# Use EUI instead of absolute meter readings.
data.loc[:,'eui'] = data.loc[:, 'meter_reading']/data.loc[:,'sqm']

# Remove outliers.
q1 = data.loc[:,'eui'].quantile(0.25, interpolation='midpoint')
q3 = data.loc[:,'eui'].quantile(0.75, interpolation='midpoint')
iqr = q3 - q1

upper = q3 + 1.5*iqr
lower = q1 - 1.5*iqr

upper_mask = data.loc[:,'eui'] > upper
lower_mask = data.loc[:,'eui'] < lower

data.drop(index=data.index[upper_mask | lower_mask], inplace=True)

data.loc[:,'simpleusage'] = data.loc[:,'primaryspaceusage']
data.loc[~data.loc[:,'simpleusage'].isin(["Education", "Office"]),'simpleusage']='Other'

cats_in_model = ['simpleusage'] # 'site_id', 

### CREATE SEPARATE MODEL FOR OFFICE, EDUCATION, AND OTHER? ###

X = data.loc[:, cats_in_model+['hdd', 'cdd']]
y = data.loc[:,['eui']]

X_numeric = X.select_dtypes(include=np.number)
scaler = StandardScaler().fit(X_numeric)
X_scaled = scaler.transform(X_numeric)
# normalizer = Normalizer().fit_transform(X_scaled, y=None, fit_params= dict(axis=0, norm='l2'))
# data_norm = normalizer.transform(X_scaled)
# data_norm = pd.DataFrame(data_norm, columns = X_numeric.columns, index=X_numeric.index)
X_scaled = pd.DataFrame(X_scaled, columns = X_numeric.columns, index=X_numeric.index)
X_scaled = pd.concat([X.select_dtypes(exclude=np.number), X_scaled], axis=1)

# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

train_mask = data.loc[:,'timestamp'].dt.year == 2016
test_mask = data.loc[:,'timestamp'].dt.year == 2017

X_train = X_scaled.loc[train_mask, :]
y_train = y.loc[train_mask, :]
X_test = X_scaled.loc[test_mask, :]
y_test = y.loc[test_mask, :]

# One-Hot Encoding:
encoder_one_hot = OneHotEncoder()
X_train_one_hot = encoder_one_hot.fit_transform(X_train[cats_in_model])

X_train_no_cat = X_train.drop(columns=cats_in_model)

# Build linear regression model
model_one_hot = LinearRegression().fit(X_train_one_hot, y_train)
model_no_cat = LinearRegression().fit(X_train_no_cat, y_train)

X_test_one_hot = encoder_one_hot.transform(X_test[cats_in_model])
y_pred_one_hot = model_one_hot.predict(X_test_one_hot)

X_test_no_cat = X_test.drop(columns=cats_in_model)
y_pred_no_cat = model_no_cat.predict(X_test_no_cat)

rmse_one_hot = root_mean_squared_error(y_test, y_pred_one_hot)
mae_one_hot = median_absolute_error(y_test, y_pred_one_hot)
print(f"One-Hot Encoding Model - Root Mean Squared Error: {rmse_one_hot}")
print(f"One-Hot Encoding Model - Mean Absolute Error: {mae_one_hot}")

rmse_no_cat = root_mean_squared_error(y_test, y_pred_no_cat)
mae_no_cat = median_absolute_error(y_test, y_pred_no_cat)
print(f"One-Hot Encoding Model - Root Mean Squared Error: {rmse_one_hot}")
print(f"One-Hot Encoding Model - Mean Absolute Error: {mae_one_hot}")

# model_one_hot.coef_
# model_one_hot.intercept_
print('END')