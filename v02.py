#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 22:02:56 2019

@author: pasi
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GroupShuffleSplit

X = np.load('X_train_kaggle.npy')
X_test = np.load('X_test_kaggle.npy')

groupsdata = []

with open("groups.csv") as f:
    for line in f:
        values = line.split(",")
        values = [v.rstrip() for v in values]
        groupsdata.append(values)
        
groupsdata = np.array(groupsdata)

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

print(groupsdata[:,2])
groupsdata[:,2] = labelencoder.fit_transform(groupsdata[:,2])
print(groupsdata[:,2])

groupsdata = groupsdata[1:,:].astype(int)

ind = groupsdata[:,0]
groups = groupsdata[:,1]
y = groupsdata[:,2]

X_means = X.mean(2)
x_vars = X.var(2)

X = np.concatenate((X_means, x_vars), axis=1)
X = np.absolute(X[:,[4,5,6,7,8,9,14,15,16,17]])

from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
#X = normalize(X, axis=1)

gss = GroupShuffleSplit(random_state=1);


for train, test in gss.split(X, y, groups=groups):
    #print("%s %s" % (train, test))
    for classifier in [(KNeighborsClassifier, KNeighborsClassifier()), (LinearDiscriminantAnalysis ,LinearDiscriminantAnalysis()), (SVC, SVC(gamma="auto")), (LogisticRegression ,LogisticRegression(max_iter=1000, class_weight="balanced", penalty="l2", multi_class="auto", solver="lbfgs"))]:
        model = classifier[1]
        model.fit(np.take(X, train, axis=0), np.take(y, train))
        pred = model.predict(np.take(X, test, axis=0))
        print("%s %s" % (accuracy_score(np.take(y, test), pred), classifier[0]))
    print()

model = SVC(gamma="auto")
model.fit(X,y)
X_test_means = X_test.mean(2)
X_test_vars = X_test.var(2)
X_test = np.concatenate((X_test_means, X_test_vars), axis=1)
X_test = X_test[:,[4,5,6,7,8,9,14,15,16,17]]
X_test = scaler.fit_transform(X_test)

y_pred = model.predict(X_test)
print(y_pred.astype(str))
labels = list(labelencoder.inverse_transform(y_pred))
print(labels)
with open("submission.csv", "w") as fp:
    fp.write("# Id,Surface\n")
    for i, label in enumerate(labels): 
        fp.write("%d,%s\n" % (i, label))
        