# -*- coding: utf-8 -*-
"""Diabetes-Prediction.ipynb

Importing the dependencies
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

"""Data Collection & Analysis

PIMA Diabetes dataset
"""

# Loading the diabetes Dataset
diabetes_dataset = pd.read_csv('/content/diabetes.csv')

pd.read_csv?

#Printing the first few rows of dataset
diabetes_dataset.head()

#Number of Rows & Columns In this Dataset
diabetes_dataset.shape

diabetes_dataset.describe()

diabetes_dataset['Outcome'].value_counts()

"""0 represents Non-Diabetic
1 represents diabetic
"""

diabetes_dataset.groupby('Outcome').mean()

#Separating the Data & Labels
X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']

print(X)

print(Y)

"""Data Standardization

"""

scaler = StandardScaler()

scaler.fit(X)

standardized_data = scaler.transform(X)

print(standardized_data)

X = standardized_data
Y = diabetes_dataset['Outcome']

print(X)
print(Y)

"""Train Test Split

"""

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify = Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

"""Training the Model"""

classifier = svm.SVC(kernel='linear')

#Training the Support Vector
classifier.fit(X_train, Y_train)

"""Model Evaluation &
Accuracy Score
"""

X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy score of the training data is: ' , training_data_accuracy)

#On test Data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score for the test data: ', test_data_accuracy)

"""Making a predictive system"""

input_data = (4,110,92,0,0,37.6,0.191,30) 

#changing the data to numpy array

input_data_as_numpy_array = np.asarray(input_data)

# Reshape the array as we are expecting one instance

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# Standardize the input data

std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)

print(prediction)


if(prediction[0]==0):  
  print('The person is not Dibetic')
else:
  print('the person is Diabetic')

# the [0] because it return a list and in python indexing starts from 0