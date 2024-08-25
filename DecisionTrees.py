# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 21:41:47 2024

@author: furko
"""

import pandas as pd

data = pd.read_csv("DecisionTreesDataSet.csv")
data['Sex'] = pd.factorize(data['Sex'])[0] + 1
data['BP'] = pd.factorize(data['BP'])[0] + 1
data['Cholesterol'] = pd.factorize(data['Cholesterol'])[0] + 1
data['Drug'] = pd.factorize(data['Drug'])[0] + 1

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(data)
data_scaled = scaler.transform(data)

data_scaled = pd.DataFrame(data_scaled, columns=data.columns)

from sklearn.model_selection import train_test_split

train, test = train_test_split(data, train_size=0.2, random_state=53)
train_result = train.iloc[:,-1]
test_result = test.iloc[:,-1]

train = train.drop(["Drug"], axis=1)
test = test.drop(["Drug"], axis=1)

from sklearn.tree import DecisionTreeClassifier

dpt = 4
tree = DecisionTreeClassifier(criterion="entropy", max_depth=dpt)
tree.fit(train, train_result)

from sklearn import metrics
print("Depth value:", dpt)
print("Train set Accuracy: ", metrics.accuracy_score(train_result, tree.predict(train)))
print("Test set Accuracy: ", metrics.accuracy_score(test_result, tree.predict(test)))
