# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from datetime import timedelta
from sklearn.preprocessing import LabelEncoder


customers = pd.read_csv('ml_case_training_data.csv')
prices = pd.read_csv('ml_case_training_hist_data.csv')


#####################   examine customer data   ##########################


print(customers.describe())
print(customers.info())

#### turning dates to datetime objects

customers['date_activ'] = pd.to_datetime(customers['date_activ'], format = '%Y-%m-%d')
customers['date_end'] = pd.to_datetime(customers['date_end'], format = '%Y-%m-%d')
customers['date_modif_prod'] = pd.to_datetime(customers['date_modif_prod'], format = '%Y-%m-%d')
customers['date_renewal'] = pd.to_datetime(customers['date_renewal'], format = '%Y-%m-%d')


#### examine number of null values per column
print(customers.isna().sum())



### drop colummns with many nulls
columns_to_drop = ['activity_new','campaign_disc_ele','date_first_activ','forecast_base_bill_ele',
                   'forecast_base_bill_year','forecast_bill_12m','forecast_cons']


customers = customers.drop(columns = columns_to_drop)


### replace null values for columns that we didn't drop
customers['channel_sales'] = customers['channel_sales'].fillna('foosdfpfkusacimwkcsosbicdxkicaua')
customers.loc[customers['date_modif_prod'].isnull(),'date_modif_prod'] = customers['date_modif_prod'].value_counts().index[0]
customers.loc[customers['date_end'].isnull(),'date_end'] = customers['date_end'].value_counts().index[0]
customers.loc[customers['date_renewal'].isnull(),'date_renewal'] = customers['date_renewal'].value_counts().index[0]
customers['forecast_discount_energy'] = customers['forecast_discount_energy'].fillna(0)
customers['forecast_price_energy_p1'] = customers['forecast_price_energy_p1'].fillna(customers['forecast_price_energy_p1'].mean())
customers['forecast_price_energy_p2'] = customers['forecast_price_energy_p2'].fillna(customers['forecast_price_energy_p2'].mean())
customers['forecast_price_pow_p1'] = customers['forecast_price_pow_p1'].fillna(customers['forecast_price_pow_p1'].mean())
customers['margin_gross_pow_ele'] = customers['margin_gross_pow_ele'].fillna(customers['margin_gross_pow_ele'].mean())
customers['margin_net_pow_ele'] = customers['margin_net_pow_ele'].fillna(customers['margin_net_pow_ele'].mean())
customers['net_margin'] = customers['net_margin'].fillna(customers['net_margin'].mean())
customers['origin_up'] = customers['origin_up'].fillna('lxidpiddsbxsbosboudacockeimpuepw')
customers['pow_max'] = customers['pow_max'].fillna(customers['pow_max'].mean())


### encode categorical features
lencoder = LabelEncoder()
lencoder.fit(customers['channel_sales'])
customers['channel_sales'] = lencoder.transform(customers['channel_sales'])

lencoder2 = LabelEncoder()
lencoder2.fit(customers['origin_up'])
customers['origin_up'] = lencoder2.transform(customers['origin_up'])


lencoder3= LabelEncoder()
lencoder3.fit(customers['has_gas'])
customers['has_gas'] = lencoder3.transform(customers['has_gas'])





########################  clean historical prices data   ###################


prices['price_date'] = pd.to_datetime(prices['price_date'], format = '%Y-%m-%d')


### replace null values'
prices['price_p1_var'] = prices['price_p1_var'].fillna(prices['price_p1_var'].mean())
prices['price_p2_var'] = prices['price_p2_var'].fillna(prices['price_p2_var'].mean())
prices['price_p3_var'] = prices['price_p3_var'].fillna(prices['price_p3_var'].mean())
prices['price_p1_fix'] = prices['price_p1_fix'].fillna(prices['price_p1_fix'].mean())
prices['price_p2_fix'] = prices['price_p2_fix'].fillna(prices['price_p2_fix'].mean())
prices['price_p3_fix'] = prices['price_p3_fix'].fillna(prices['price_p3_fix'].mean())




### negative pricing values are not acceptable - have to make them zero
prices['price_p1_fix'] = prices['price_p1_fix'].apply(lambda x : 0 if x<0 else x)
prices['price_p2_fix'] = prices['price_p2_fix'].apply(lambda x : 0 if x<0 else x)
prices['price_p3_fix'] = prices['price_p3_fix'].apply(lambda x : 0 if x<0 else x)



customers.to_csv('cleaned_customers.csv',encoding='utf-8',index=False)
prices.to_csv('cleaned_prices.csv',encoding='utf-8',index=False)



