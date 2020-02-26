#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 15:16:50 2019

@author: z003zhj
"""

import numpy as np
from sklearn import datasets
iris= datasets.load_iris()
iris_x= iris.data
iris_y= iris.target
np.unique(iris_y)
np.random.seed(0)
indices=np.random.permutation(len(iris_x))
iris_x_train= iris_x[indices[:-10]]
iris_y_train= iris_y[indices[:-10]]
iris_x_test= iris_x[indices[-10:]]
iris_y_test= iris_y[indices[-10:]]
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(iris_x_train, iris_y_train)
print(knn.predict(iris_x_test))
print(iris_y_test)



#linear regression
import numpy as np
from sklearn import datasets
di= datasets.load_diabetes()
diabetes_x_train = di.data[:-20]
diabetes_y_train = di.target[:-20]
diabetes_x_test = di.data[-20:]
diabetes_y_test = di.target[-20:]
from sklearn import linear_model
regr= linear_model.LinearRegression()
regr.fit(diabetes_x_train, diabetes_y_train)
print(regr.coef_)
print(np.mean((regr.predict(diabetes_x_test)-diabetes_y_test)**2))
print(regr.score(diabetes_x_test, diabetes_y_test))


import pandas as pd
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data' 
col_names= ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris= pd.read_csv(url, header=None, names=col_names)
iris_class = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}
iris['species_num'] = [iris_class[i] for i in iris.species]
x= iris.drop(['species', 'species_num'], axis=1)
y= iris.species_num

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=42)
from sklearn.neighbors import KNeighborsClassifier
clf= KNeighborsClassifier(n_neighbors=5)
clf.fit(x_train, y_train)
clf.score(x_test, y_test)

                                                
from sklearn.datasets import load_iris
iris= load_iris()
print(type(iris))
print(iris.data)
print(iris.feature_names)
print(iris.target)
print(iris.target_names)
print(iris.data.shape)
x=iris.data
y=iris.target
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=4)
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
k_range= range(1,26)
scores= {}
scores_list =[]
for k in k_range:
    knn= KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred=knn.predict(x_test)
    scores[k]= metrics.accuracy_score(y_test, y_pred)
    scores_list.append(metrics.accuracy_score(y_test, y_pred))
    
import matplotlib.pyplot as plt
plt.plot(k_range, scores_list)
plt.xlabel('value of k for knn')
plt.ylabel('testing accuracy')

    

from sklearn.datasets import fetch_20newsgroups
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups()
#twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True , random_state=42)
#twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True , random_state=42)
print(twenty_train.data)
print(twenty_train.filenames)
print(twenty_train.targetnames)



    






