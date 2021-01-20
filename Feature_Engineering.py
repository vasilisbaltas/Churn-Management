# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from datetime import timedelta
from sklearn.preprocessing import LabelEncoder
from scipy.stats import zscore



customers = pd.read_csv('cleaned_customers.csv')
prices = pd.read_csv('cleaned_prices.csv')
churn = pd.read_csv('ml_case_training_output.csv')


print(prices.price_date.max())
print(prices.price_date.min())



customers['date_activ'] = pd.to_datetime(customers['date_activ'], format = '%Y-%m-%d')
customers['date_end'] = pd.to_datetime(customers['date_end'], format = '%Y-%m-%d')
customers['date_modif_prod'] = pd.to_datetime(customers['date_modif_prod'], format = '%Y-%m-%d')
customers['date_renewal'] = pd.to_datetime(customers['date_renewal'], format = '%Y-%m-%d')


df = pd.merge(customers, churn, on='id')


df['tenure'] = ((df["date_end"]-df["date_activ"])/ np.timedelta64(1, "Y")).astype(int)




#### a way of transforming dates to months duration

def convert_months(reference_date, dataframe, column):
    time_delta = reference_date - dataframe[column]
    months = (time_delta / np.timedelta64(1, 'M')).astype(int)
    
    
    return months



REFERENCE_DATE = datetime.datetime(2016, 1, 1)

df['months_since_activ'] = convert_months(REFERENCE_DATE, df, 'date_activ')
df['months_to_end'] = -convert_months(REFERENCE_DATE, df, 'date_end')
df['months_since_modif'] = convert_months(REFERENCE_DATE, df, 'date_modif_prod')
df['months_since_renew']  = convert_months(REFERENCE_DATE, df, 'date_renewal')




### drop unnecessary columns

columns_to_drop = ['date_activ', 'date_modif_prod', 'date_renewal', 'date_end']
    
df = df.drop(columns = columns_to_drop)    
    



### log-transform for skewed variables


df.loc[df.cons_12m < 0,"cons_12m"] = np.nan
df.loc[df.cons_gas_12m < 0,"cons_gas_12m"] = np.nan
df.loc[df.cons_last_month < 0,"cons_last_month"] = np.nan
df.loc[df.forecast_cons_12m < 0,"forecast_cons_12m"] = np.nan
df.loc[df.forecast_cons_year < 0,"forecast_cons_year"] = np.nan
df.loc[df.forecast_meter_rent_12m < 0,"forecast_meter_rent_12m"] = np.nan
df.loc[df.imp_cons < 0,"imp_cons"] = np.nan


df["cons_12m"] = np.log10(df["cons_12m"]+1)
df["cons_gas_12m"] = np.log10(df["cons_gas_12m"]+1)
df["cons_last_month"] = np.log10(df["cons_last_month"]+1)
df["forecast_cons_12m"] = np.log10(df["forecast_cons_12m"]+1)
df["forecast_cons_year"] = np.log10(df["forecast_cons_year"]+1)
df["forecast_meter_rent_12m"] = np.log10(df["forecast_meter_rent_12m"]+1)
df["imp_cons"] = np.log10(df["imp_cons"]+1)





### replace outliers with Z-score

def replace_outliers_z_score(dataframe, column, Z=3):
    
    df = dataframe.copy(deep=True)
    df.dropna(inplace=True, subset=[column])
    
    df["zscore"] = zscore(df[column])
    mean_ = df[(df["zscore"] > -Z) & (df["zscore"] < Z)][column].mean()
    
    no_outliers = dataframe[column].isnull().sum()
    dataframe[column] = dataframe[column].fillna(mean_)
    dataframe["zscore"] = zscore(dataframe[column])
    dataframe.loc[(dataframe["zscore"] < -Z) | (dataframe["zscore"] > Z),column] = mean_
    
    
    print("Replaced:", no_outliers, " outliers in ", column)
    return dataframe.drop(columns="zscore")




df = replace_outliers_z_score(df,"cons_12m")
df = replace_outliers_z_score(df,"cons_gas_12m")
df = replace_outliers_z_score(df,"cons_last_month")
df = replace_outliers_z_score(df,"forecast_cons_12m")
df = replace_outliers_z_score(df,"forecast_cons_year")
df = replace_outliers_z_score(df,"forecast_discount_energy")
df = replace_outliers_z_score(df,"forecast_meter_rent_12m")
df = replace_outliers_z_score(df,"forecast_price_energy_p1")
df = replace_outliers_z_score(df,"forecast_price_energy_p2")
df = replace_outliers_z_score(df,"forecast_price_pow_p1")
df = replace_outliers_z_score(df,"imp_cons")
df = replace_outliers_z_score(df,"margin_gross_pow_ele")
df = replace_outliers_z_score(df,"margin_net_pow_ele")
df = replace_outliers_z_score(df,"net_margin")
df = replace_outliers_z_score(df,"pow_max")
df = replace_outliers_z_score(df,"months_since_activ")
df = replace_outliers_z_score(df,"months_to_end")
df = replace_outliers_z_score(df,"months_since_modif")
df = replace_outliers_z_score(df,"months_since_renew")


df.reset_index(drop=True, inplace=True)

    


#####   averaging prices



df['avg_price_p1_var'] = 0
df['avg_price_p2_var'] = 0
df['avg_price_p3_var'] = 0
df['avg_price_p1_fix'] = 0
df['avg_price_p2_fix'] = 0
df['avg_price_p3_fix'] = 0


for ID in pd.unique(df.id):
    
    df.loc[df.id== ID,'avg_price_p1_var'] = prices[prices.id == ID]['price_p1_var'].mean()
    df.loc[df.id== ID,'avg_price_p2_var'] = prices[prices.id == ID]['price_p2_var'].mean()
    df.loc[df.id== ID,'avg_price_p3_var'] = prices[prices.id == ID]['price_p3_var'].mean()
    df.loc[df.id== ID,'avg_price_p1_fix'] = prices[prices.id == ID]['price_p1_fix'].mean()
    df.loc[df.id== ID,'avg_price_p2_fix'] = prices[prices.id == ID]['price_p2_fix'].mean()
    df.loc[df.id== ID,'avg_price_p3_fix'] = prices[prices.id == ID]['price_p3_fix'].mean()
    
    
    
    
df.to_csv('ML_Features.csv',encoding='utf-8',index=False)    
    
    
    
    
    
    