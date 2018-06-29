#Import Libraries
# coding=utf8
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import data
dataset = pd.read_csv(‘Data.csv’)
x = dataset.iloc[:,:-1].values
y =dataset.iloc[:,1].values

#Splitting training set and testing set
from sklearn.cross_validation import train_test_split
xtrain, xtest, ytrain, ytest =train_test_split(x,y,test_size=0.25)

#Training and Fitting model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)

#Predicting using the Model

y_prediction = regressor.predict(xtest)

#Visualising the Training set Results
plt.scatter(xtrain, ytrain,color=’red’)
plt.plot(xtrain,regressor.predict(xtrain), color=’blue’)
plt.title(‘Bank Account Balance vs Years of Work (Training set)’)
plt.xlabel(‘ Years of Work’)
plt.ylabel(‘Bank Account Balance’)
plt.show

#Visualise Predicted result
plt.scatter(xtest, y_prediction,color=’red’)
plt.plot(xtrain,regressor.predict(xtrain), color=’blue’)
plt.title(‘Bank Account Balance vs Years of Work (Predicted Result)’)
plt.xlabel(‘Years of Work’)
plt.ylabel(‘Bank Account Balance’)
plt.show