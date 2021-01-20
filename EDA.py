# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from datetime import timedelta
from sklearn.preprocessing import LabelEncoder


sns.set(color_codes=True)
pd.set_option('display.max_columns',100)





customers = pd.read_csv('ml_case_training_data.csv')
churn = pd.read_csv('ml_case_training_output.csv')

df = pd.merge(customers, churn, on='id')



#### Percentage of missing values per feature
print(pd.DataFrame({'Missing values(%)' : df.isnull().sum()/len(df.index)*100}))



################################   VIZ   #####################################


activity = df [['id','activity_new','churn']]
activity = activity.groupby([activity['activity_new'],activity['churn']])['id'].count().unstack(level=1).sort_values(by=[0], ascending=False)

activity.plot(kind='bar', figsize=(18,10), width=2, stacked=True, title='SME activity')
plt.ylabel('Number of Companies')
plt.xlabel('Activity')

plt.legend(['Retention','Churn'], loc='upper right')
plt.xticks([])    #### !!! do not show names in the x-axis
plt.show()




### plotting electricity and gas consumption  - need log-transform
plt.figure(figsize=(12,8))
plt.subplot(3,1,1);  sns.distplot(customers['cons_12m'])
plt.subplot(3,1,2);  sns.distplot(customers['cons_gas_12m'])
plt.subplot(3,1,3);  sns.distplot(customers['cons_last_month'])
plt.show()


plt.figure(figsize=(12,8))
plt.subplot(2,1,1);  sns.distplot(customers['forecast_cons_12m'])
plt.subplot(2,1,2);  sns.distplot(customers['forecast_cons_year'])
plt.show()


##############################################################################



def plot_distribution(dataframe, column, ax, bins_=50):
    
    temp = pd.DataFrame({'Retention':dataframe[dataframe['churn']==0][column],
                         'Churn':dataframe[dataframe['churn']==0][column]})
    
    
    temp[['Retention','Churn']].plot(kind='hist',bins=bins_, ax=ax, stacked =True)
    ax.set_xlabel(column)
    ax.ticklabel_format(style = 'plain', axis='x')
    
    

consumption = df[['id','cons_12m','cons_gas_12m','cons_last_month','imp_cons','has_gas','churn']]
    
    
fig, axs = plt.subplots(nrows=4, figsize=(18,25))    
    
plot_distribution(consumption, 'cons_12m', axs[0])
plot_distribution(consumption[consumption['has_gas']=='t'], 'cons_gas_12m', axs[1])
plot_distribution(consumption, 'cons_last_month', axs[2])
plot_distribution(consumption, 'imp_cons', axs[3])




#############################################################################
dates = df[['id','date_activ','date_end','date_modif_prod','date_renewal','churn']].copy()


dates['date_activ'] = pd.to_datetime(dates['date_activ'], format = '%Y-%m-%d')
dates['date_end'] = pd.to_datetime(dates['date_end'], format = '%Y-%m-%d')
dates['date_modif_prod'] = pd.to_datetime(dates['date_modif_prod'], format = '%Y-%m-%d')
dates['date_renewal'] = pd.to_datetime(dates['date_renewal'], format = '%Y-%m-%d')


def plot_dates(dataframe, column, fontsize_=12):
    
    temp = dataframe[[column, 'churn', 'id']].set_index(column).groupby([pd.Grouper(freq='M'), 'churn']).count().unstack(level=1)
    
    
    ax= temp.plot(kind='bar', stacked=True, figsize=(18,10), rot=0)
    ax.set_xticklabels(map(lambda x: line_format(x), temp.index))
    
    plt.xticks(fontsize = fontsize_)
    
    plt.ylabel('Number of Companies')
    plt.legend(['Retention','Churn'], loc = 'upper right')
    plt.show()
    
    
    
def line_format(label):

   month = label.month_name()[:1]
   if label.month_name() == 'January':
       month += f'\n{label.year}'
       
   return month





plot_dates(dates, 'date_activ', fontsize_=8)    
plot_dates(dates, 'date_end')
plot_dates(dates,'date_modif_prod', fontsize_= 8)
plot_dates(dates, 'date_renewal')
       
       
    
















