#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn.metrics

from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.dummy import DummyClassifier
from matplotlib import pyplot
from sklearn.model_selection import KFold

from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
    
from matplotlib.legend_handler import HandlerLine2D



def estimatorFun(x_train,y_train, x_test, y_test ):
    n_estimators  = [1, 2, 4, 8, 16, 32, 64, 100, 200]
    train_results = []
    test_results  = []

    for estimator in n_estimators:
        rf = RandomForestClassifier(n_estimators=estimator, n_jobs=-1)
        rf.fit(x_train, y_train)
        train_pred = rf.predict(x_train)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        train_results.append(roc_auc)
        y_pred = rf.predict(x_test)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        test_results.append(roc_auc)

    line1, = plt.plot(n_estimators, train_results, "b", label="Train AUC")
    line2, = plt.plot(n_estimators, test_results, "r", label="Test AUC")
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel("AUC score")
    plt.xlabel("n_estimators")
    plt.show()

def main():
    
    feature_cols = []
    for col in df.columns: 
        if col != "from" and col != "event" and col != " status" and col != "from-time" and col != "from-pub-uri" and col != "Country" and col != "class" and col != "Class" and col != "status" and col != "1-Cosine":
            feature_cols.append(col)

        target_var = "class"

    x = df[feature_cols].values
    y = df[target_var].values
    labelencoder_X = LabelEncoder()

    for i in range(len(feature_cols)):
        x[:,i] = labelencoder_X.fit_transform(x[:,i])
    
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    
    for train_ix, test_ix in kfold.split(x, y):
        x_train, x_test = x[train_ix], x[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]
        
    estimatorFun(x_train,y_train, x_test, y_test )
    

    

