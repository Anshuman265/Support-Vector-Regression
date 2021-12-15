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

#Training the SVR model on the whole dataset
from sklearn.svm import SVR
# Choosing the radial basis function for kernel here
regressor = SVR(kernel= 'rbf')
regressor.fit(X,y) 

#Predicting a new result
print(sc_Y.inverse_transform([regressor.predict(sc_X.transform([[6.5]]))]))

#Visualising the SVR results
plt.scatter(sc_X.inverse_transform(X),sc_Y.inverse_transform(y),color = 'red')
plt.plot(sc_X.inverse_transform(X),sc_Y.inverse_transform(regressor.predict(X).reshape(-1,1)),color = 'blue')
plt.title('Support vector regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR Results (for higher resolution and smoother curve)
X_grid = np.arange(min(sc_X.inverse_transform(X)),max(sc_X.inverse_transform(X)),0.1)
X_grid = X_grid.reshape((len(X_grid),1))    
plt.scatter(sc_X.inverse_transform(X),sc_Y.inverse_transform(y),color = 'red')
plt.plot(X_grid,sc_Y.inverse_transform(regressor.predict(sc_X.transform(X_grid)).reshape(-1,1)),color = 'blue')
plt.title('Nicer resolution curve')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()




