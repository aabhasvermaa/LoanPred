# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 13:08:10 2021

@author: Deepak Baghel
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import seaborn as sns

dataset = pd.read_csv("E:\VS Workspace\Python\DATASETS.csv")
print(dataset.head())

print(dataset.isnull().sum())

dataset = dataset.dropna()

print(dataset.isnull().sum())

dataset.replace({"Loan_Status": {'N': 0, 'Y': 1}}, inplace=True)

dataset = dataset.replace(to_replace='3+', value=4)

dataset.replace({"Married": {'No': 0, 'Yes': 1}, "Gender": {'Female': 0, 'Male': 1}, "Self_Employed": {'No': 0, 'Yes': 1},
                "Property_Area": {'Rural': 0, 'Semiurban': 1, 'Urban': 2}, "Education": {'Graduate': 1, 'Not Graduate': 0}}, inplace=True)


X = dataset.drop(columns=['Loan_ID', 'Loan_Status'], axis=1)
y = dataset['Loan_Status']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=0)
display(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

print(X_train)
print(y_train)
print(X_test)
print(y_test)


mylr = LogisticRegression()

mylr.fit(X_train, y_train)
print("Logistic Regression model is build !!")

y_pred = mylr.predict(X_test)
X_pred = mylr.predict(X_train)
print(accuracy_score(X_pred, y_train))
print(accuracy_score(y_pred, y_test))

#print ("Actual testing data of loan status:", y_test)
#print ("Predicted data of loan status:",y_pred)

input_data1 = (1, 1, 1, 1, 0, 3076, 1500, 126, 360, 1, 2)

k = 1

input_data = (k, k, k, k, k, 3076, 1500, 126, 360, k, k)


input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

New_pred = mylr.predict(input_data_reshaped)

print("This is new pridiction ", New_pred)
