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


sns.set( color_codes=True )
pd.set_option('display.max_columns',100)

data = pd.read_csv('ML_Features.csv')



### checking our datasets features

print(pd.DataFrame({'Dataframe columns':data.columns}))

y = data['churn']
X = data.drop(labels = ['id','churn'], axis= 1)

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.2, random_state = 33)




### an initial model

model = xgb.XGBClassifier( learning_rate = 0.1, max_depth = 6, n_estimators = 500, n_jobs = -1)
result = model.fit(X_train, y_train)



### Model Evaluation

def evaluate( model_, X_test_, y_test_ ):
    
    ### evaluating accuracy, precision and recall of the model
    
    prediction_test_ = model_.predict(X_test_)
    
    results = pd.DataFrame({'Accuracy' : [metrics.accuracy_score(y_test_, prediction_test_)],
                            'Precision': [metrics.precision_score(y_test_, prediction_test_)],
                            'Recall': [metrics.recall_score(y_test_, prediction_test_)]})
    
    return results



evaluate(model, X_test, y_test)




### estimate the ROC curve

def calculate_roc_auc( model_, X_test_, y_test_ ):
    
    
    ### obtain the prediction for class 1 -->churn
    prediction_test_ = model_.predict_proba(X_test_)[:,1]
    
    fpr, tpr, thresholds = metrics.roc_curve(y_test_, prediction_test_)
    
    score = pd.DataFrame({'ROC-AUC' : [metrics.auc(fpr,tpr)]})
    
    return fpr, tpr, score




def plot_roc_auc( fpr, tpr ):
    
    f, ax = plt.subplots(figsize = (14,8))
    
    
    roc_auc = metrics.auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=2, alpha=0.3, label = 'AUC = %0.2f' % (roc_auc))
    
    plt.plot([0,1], [0,1], linestyle = '--', lw=3, color = 'r', label='Random', alpha=.8)
    
    
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.set_title('ROC-AUC')
    ax.legend(loc = 'lower right')
    
    plt.show()
    
    
    

fpr, tpr, auc_score = calculate_roc_auc(model, X_test, y_test)
print(auc_score)
    

plot_roc_auc(fpr, tpr)
plt.show()
    




### Stratified K-fold validation
    
# this function plots the receiver operating characteristic curve from a list of 
# true positive rates and false positive rates 
   
def plot_roc_curve(fprs, tprs):
    
    tprs_interp = []
    aucs = []
    
    mean_fpr = np.linspace(0,1,100)
    
    f, ax = plt.subplots(figsize = (18,10))
    
    
    ### plot ROC for each fold and compute auc scores
    
    for i, (fpr,tpr) in enumerate(zip(fprs,tprs)):
        
        tprs_interp.append(np.interp(mean_fpr, fpr, tpr))
        tprs_interp[-1][0] = 0.0
        
        roc_auc = metrics.auc(fpr,tpr)
        aucs.append(roc_auc)
        
        ax.plot(fpr, tpr, lw=2, alpha=0.3, label = 'ROC fold %d (AUC = %0.2f)' % (i,roc_auc))
        
        
    plt.plot([0,1], [0,1], linestyle = '--', lw=3, color='r', label = 'Random', alpha=.8)  
    
    
    
    ### plot the mean ROC
    
    mean_tpr = np.mean(tprs_interp, axis = 0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    
    ax.plot(mean_fpr, mean_tpr, color='b', label = r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
            lw =4, alpha = .8)
    
    
    ### plot standard deviation around the mean ROC
    
    std_tpr = np.std(tprs_interp, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey',alpha=.2, label = r"$\pm$ 1 std. dev.")
    
    
    
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.set_title('ROC-AUC')
    ax.legend(loc = 'lower right')
    
    plt.show()
 
    return(f, ax)



def compute_roc_auc(model_, index):
    
    y_predict = model_.predict_proba(X.iloc[index])[:,1]
    fpr, tpr, thresholds = metrics.roc_curve(y.iloc[index], y_predict)
    auc_score = metrics.auc(fpr,tpr)
    
    return fpr,tpr,auc_score




cv = StratifiedKFold(n_splits = 5, random_state = 33, shuffle = True)

fprs, tprs, scores = [], [], []

for (train,test),i in zip(cv.split(X,y), range(5)):
    
    model.fit(X.iloc[train], y.iloc[train])
    _, _, auc_score_train = compute_roc_auc(model, train)
    fpr, tpr, auc_score = compute_roc_auc(model, test)
    
    scores.append((auc_score_train, auc_score))
    fprs.append(fpr)
    tprs.append(tpr)
    
    

plot_roc_curve(fprs,tprs)
plt.show()








