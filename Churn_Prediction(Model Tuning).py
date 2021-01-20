# -*- coding: utf-8 -*-


import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from Churn_Prediction import compute_roc_auc, plot_roc_curve
from sklearn.inspection import plot_partial_dependence


data = pd.read_csv('ML_Features.csv')
y = data['churn']
X = data.drop(labels = ['id','churn'], axis= 1)

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.2, random_state = 33)



### create the random grid

params = {
          'min_child_weight': [ i for i in np.arange(1,15,1)],
          'gamma': [ i for i in np.arange(0,6,0.5)],
          'subsample': [ i for i in np.arange(0,1.1,0.1)],
          'colsample_bytree': [ i for i in np.arange(0,1.1,0.1)],
          'max_depth': [ i for i in np.arange(1,15,1)],
          'scale_pos_weight': [ i for i in np.arange(1,15,1)],
          'learning_rate': [ i for i in np.arange(0,0.15,0.01)],
          'n_estimators': [ i for i in np.arange(0,2000,100)]
          }


### our model

xg = xgb.XGBClassifier(objective = 'binary:logistic', silent=True, nthread=1)


### apply rendom search

xg_random = RandomizedSearchCV(xg, param_distributions = params, n_iter=200, scoring = 'roc_auc',
                               n_jobs= -1, cv=5, verbose = 2, random_state = 33)

xg_random.fit(X_train, y_train)


best_random = xg_random.best_params_


### create a model with the parameters found

model_random = xgb.XGBClassifier(objective = 'binary:logistic', silent=True, nthread=1, **best_random)



cv = StratifiedKFold(n_splits = 5, random_state = 33, shuffle = True)

fprs, tprs, scores = [], [], []

for (train,test),i in zip(cv.split(X,y), range(5)):
    
    model_random.fit(X.iloc[train], y.iloc[train])
    _, _, auc_score_train = compute_roc_auc(model_random, train)
    fpr, tpr, auc_score = compute_roc_auc(model_random, test)
    
    scores.append((auc_score_train, auc_score))
    fprs.append(fpr)
    tprs.append(tpr)
    

plot_roc_curve(fprs, tprs)
plt.show()




### we can also apply a grid search -- after some random experimentation with parameter values
### we can reduce the size of the grid

param_grid = {
              'subsample':[0.6,0.7],
              'scale_pos_weight': [1],
              'n)estimators': [1100],
              'min_child_weight':[1],
              'max_depth': [12,13,14],
              'learning_rate': [0.005, 0.01],
              'gamma': [4],
              'colsample_bytree': [0.5,0.6]
              }



### the second model

xg_2 = xgb.XGBClassifier(objective = 'binary:logistic', silent=True, nthread=1)


grid_search = GridSearchCV(estimator = xg_2, param_grid = param_grid, cv=5, n_jobs = -1, verbose=2, scoring = 'roc_auc')

grid_search.fit(X_train, y_train)

best_grid = grid_search.best_params_

model_grid = xgb.XGBClassifier(objective = 'binary:logistic', silent=True, nthread=1, **best_grid)


fprs, tprs, scores = [], [], []

for (train,test),i in zip(cv.split(X,y), range(5)):
    
    model_grid.fit(X.iloc[train], y.iloc[train])
    _, _, auc_score_train = compute_roc_auc(model_grid, train)
    fpr, tpr, auc_score = compute_roc_auc(model_grid, test)
    
    scores.append((auc_score_train, auc_score))
    fprs.append(fpr)
    tprs.append(tpr)
    

plot_roc_curve(fprs, tprs)
plt.show()



### understanding the model by inspecting feature importance

fig, ax = plt.subplots(figsize=(15,20))
xgb.plot_importance(model_grid, ax=ax)
plt.show()














