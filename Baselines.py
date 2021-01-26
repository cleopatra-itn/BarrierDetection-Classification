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



def evaluate_model(X, y, model):
    # define evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate model
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores

def get_models():
    models, names = list(), list()
    # Uniformly Random Guess
    models.append(DummyClassifier(strategy='uniform'))
    names.append('Uniform')
    # Prior Random Guess
    models.append(DummyClassifier(strategy='stratified'))
    names.append('Stratified')
    # Majority Class: Predict 0
    models.append(DummyClassifier(strategy='most_frequent'))
    names.append('Majority')
    # Minority Class: Predict 1
    models.append(DummyClassifier(strategy='constant', constant=1))
    names.append('Minority')
    # Class Prior
    models.append(DummyClassifier(strategy='prior'))
    names.append('Prior')
    return models, names

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
    
    models, names = get_models()
    results = list()
    for i in range(len(models)):
        scores = evaluate_model(x, y, models[i])
        #print(scores)
        results.append(scores)
        print('>%s %.3f (%.3f)' % (names[i], mean(scores), std(scores)))
    
    pyplot.boxplot(results, labels=names, showmeans=True)
    pyplot.show()

