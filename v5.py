#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 08:26:09 2019

@author: pasi
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from sklearn.naive_bayes import GaussianNB


x = np.load("X_train_kaggle.npy")

# abs values
x[:,[4,5,6,7,8,9]] = np.abs(x[:,[4,5,6,7,8,9]])

# Normalisointi
x_min = x.min(axis=(1,2), keepdims=True)
x_max = x.max(axis=(1,2), keepdims=True)
x = (x - x_min)/(x_max-x_min)

# Autokorrelattio
autocor = []
tot = []
for j in np.arange(1703):
    for i in np.arange(4,10):
        s = pd.Series(x[j,i,:])
        autocor.append(s.autocorr(lag=5))
    tot.append(autocor)
    autocor = []
tot = np.array(tot)

# Liukuva keskiarvo
ma = []
matot = []
for j in np.arange(1703):
    for i in np.arange(4,10):
        ma.append(np.ma.array(x[2,i,:]).mean())
    matot.append(ma)
    ma = []
matot = np.array(matot)

#X = x

# max ja min
X_max = x[:,[4,5,6,7,8,9]].max(2)
X_min = x[:,[4,5,6,7,8,9]].min(2)


X_means = x[:,[4,5,6,7,8,9]].mean(2)
X_stds = x[:,[4,5,6,7,8,9]].std(2)
X = np.concatenate((X_means, X_stds, X_max - X_min, tot, matot), axis=1)

gss = GroupShuffleSplit(n_splits=25,random_state=1);
res = []
restotal = []
for train, test in gss.split(X, y, groups=groups):
    #print("%s %s" % (train, test))
    
    for classifier in [(KNeighborsClassifier, KNeighborsClassifier()), (LinearDiscriminantAnalysis ,LinearDiscriminantAnalysis()), (SVC, SVC(kernel="rbf", gamma=0.1)), (SVC, SVC(kernel="linear", gamma=0.1)), (LogisticRegression ,LogisticRegression(C=200, max_iter=10000, multi_class="auto", solver="lbfgs")), (RandomForestClassifier, RandomForestClassifier(n_estimators=100)), (GaussianNB, GaussianNB()) ]:
        model = classifier[1]
        model.fit(np.take(X, train, axis=0), np.take(y, train))
        pred = model.predict(np.take(X, test, axis=0))
        #print("%s %s" % (accuracy_score(np.take(y, test), pred), classifier[0]))
        res.append((accuracy_score(np.take(y, test), pred)))
    restotal.append(np.array(res))
    res = []
    #print()
    
restotal = np.array(restotal)
restotal.mean(0)

