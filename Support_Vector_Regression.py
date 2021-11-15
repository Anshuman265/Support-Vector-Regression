#Importing the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[ : , 1:-1].values
y = dataset.iloc[ : , -1].values

#Transforming y into a 2d array
y = y.reshape(len(y),1)

#Applying feature scaling in SVR
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_Y = StandardScaler()
y  = sc_Y.fit_transform(y)