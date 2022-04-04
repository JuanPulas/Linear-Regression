import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

dirname = os.path.dirname(__file__)
stocks_file = os.path.join(dirname, 'stocks.csv')
data = pd.read_csv(stocks_file)

# Handle the NaN values
data.dropna()

X = data['High']
X = X.fillna(X.mean()) # Used fillna() because otherwise it would be inconsistent # of values between X and Y
X = X.values.reshape(-1,1) # reshaped because array was 1D instead of expected 2D

y = data['Close']
y = y.fillna(y.mean())
y = y.values.reshape(-1,1)

plt.scatter(X, y)
plt.show()
# Split
# Use scikit-learn’s train_test_split() method to split x into 80% training set and 20% testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)

plt.scatter(X_train,y_train,s=5)
plt.xlabel('X')
plt.ylabel('y')
plt.show()

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
score = lr_model.score(X_train, y_train)
print("Score: ", score)

prediction = lr_model.predict(X_test)

plt.scatter(X_test, y_test, s=5)
plt.plot(X_test, prediction, color='red')
plt.xlabel('X Test')
plt.ylabel('y Predictions')
plt.show()

sq_err = mean_squared_error(y_test, prediction)
print("Mean Square Error: ", sq_err)

