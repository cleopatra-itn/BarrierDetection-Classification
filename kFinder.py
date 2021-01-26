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
from sklearn.neighbors import KNeighborsClassifier



def findkFun(x_train, x_test, y_train, y_test):  
    neighbors = np.arange(1,9)
    train_accuracy =np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))
    for i,k in enumerate(neighbors):
        #Setup a knn classifier with k neighbors
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train, y_train)
        train_accuracy[i] = knn.score(x_train, y_train)
        test_accuracy[i] = knn.score(x_test, y_test) 

    plt.title('k-NN Varying number of neighbors')
    plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
    plt.plot(neighbors, train_accuracy, label='Training accuracy')
    plt.legend()
    plt.xlabel('Number of neighbors')
    plt.ylabel('Accuracy')
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
        
    findkFun(x_train, x_test, y_train, y_test)
    

    

