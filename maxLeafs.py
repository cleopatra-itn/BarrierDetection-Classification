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



def findMaxLeafsFun(x_train, x_test, y_train, y_test):
    for i in range(2, 100):
        decision_tree = DecisionTreeClassifier(max_leaf_nodes=i)
        decision_tree.fit(x_train, y_train)
        y_pred_test = decision_tree.predict(x_test)
        y_pred_train = decision_tree.predict(x_train)
        acc_train = accuracy_score(y_train, y_pred_train)
        acc_test = accuracy_score(y_test, y_pred_test)
        print("\t{0:5.2f}\t{1:5.2f}\t{2:5.2f}".format(decision_tree.tree_.node_count, acc_train,acc_test ))

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
        
    findMaxLeafsFun(x_train, x_test, y_train, y_test)
    

    

