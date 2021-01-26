#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
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


def classificationModelsFun(x_train, x_test, y_train, y_test):
    #Train each model and generate score. Use one at a time and comment lines of other models
    #RandomForest Classifier
    clf = RandomForestClassifier(n_estimators = 200)
    clf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)

    #Decision Tree Classifier
    decision_tree = DecisionTreeClassifier(max_leaf_nodes=63)
    decision_tree.fit(x_train, y_train)
    y_pred = decision_tree.predict(x_test)

    #SVM
    clf = SVC(kernel='linear')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    confusion_matrix(y_test,y_pred)
    pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)

    #Guassian Naive Bayes Classifier
    model = GaussianNB()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    confusion_matrix(y_test,y_pred)
    pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)

    #kNeartest Neighbors Classifier
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(x_train,y_train)
    knn.score(x_test,y_test)
    y_pred = knn.predict(x_test)
    confusion_matrix(y_test,y_pred)
    pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)


    #Score on test data set for one classifier
    acc = accuracy_score(y_test, y_pred, normalize=True)
    print('Accuracy: %.3f' % acc)

    precision = precision_score(y_test, y_pred, average='binary')
    print('Precision: %.3f' % precision)

    precision = precision_score(y_test, y_pred,  average='micro')
    print('micro Precision: %.3f' % precision)

    precision = precision_score(y_test, y_pred, average='macro')
    print('macro Precision: %.3f' % precision)

    recall = recall_score(y_test, y_pred, average='binary')
    print('Recall: %.3f' % recall)

    recall = recall_score(y_test, y_pred, average='micro')
    print('micro Recall: %.3f' % recall)

    recall = recall_score(y_test, y_pred, average='macro')
    print('macro Recall: %.3f' % recall)

    score = f1_score(y_test, y_pred, average='binary')
    print('F-Measure: %.3f' % score)

    score = f1_score(y_test, y_pred, average='micro')
    print('micro F-Measure: %.3f' % score)

    score = f1_score(y_test, y_pred, average='macro')
    print('macro F-Measure: %.3f' % score)

def main():
    feature_cols = []
    for col in df.columns: 
        if col != "from" and col != "event" and col != " status" and col != "from-time" and col != "from-pub-uri" and col != "Country" and col != "class" and col != "Class" and col != "status" and col != "1-Cosine":
            feature_cols.append(col)

        target_var = "class"

    x = df[feature_cols].values
    y = df[target_var].values

    #Do not encode for geographical and economical barrier but for all others
    labelencoder_X = LabelEncoder()
    for i in range(len(feature_cols)):
        x[:,i] = labelencoder_X.fit_transform(x[:,i])

    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    
    for train_ix, test_ix in kfold.split(x, y):
        x_train, x_test = x[train_ix], x[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]
        
    classificationModelsFun(x_train, x_test, y_train, y_test)


if __name__ == '__main__':
    main()


# In[ ]:





# In[ ]:


feature_cols = []
for col in df.columns: 
    if col != "from" and col != "event" and col != " status" and col != "from-time" and col != "from-pub-uri" and col != "Country" and col != "class" and col != "Class" and col != "status" and col != "1-Cosine":
        feature_cols.append(col)

target_var = "class"


# In[ ]:





# In[ ]:




