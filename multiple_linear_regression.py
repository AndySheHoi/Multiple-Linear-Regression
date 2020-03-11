# Multiple Linear Regression

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)


# Building the optimal model using Backward Elimination
import statsmodels.api as sm

# Adding a column [shape[0], 1] which represents constent values
X_train = np.append(arr = np.ones((X_train.shape[0], 1)), values = X_train, axis = 1)
X_test = np.append(arr = np.ones((X_test.shape[0], 1)), values = X_test, axis = 1)

# SL(Significane Level) = 0.05, any P value > SL will be eliminated
# The best model will have the highest adjusted R-squared value

# The first elimination, adjusted R-squared = 0.943
X_opt = X_train[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()

# The second elimination, adjusted R-squared = 0.944
X_opt = X_train[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()

# The third elimination, adjusted R-squared = 0.946
X_opt = X_train[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()

# The forth elimination, adjusted R-squared = 0.947
X_opt = X_train[:, [0,3,5]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()

# The fifth elimination, adjusted R-squared = 0.944
X_opt = X_train[:, [0,3]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()

# The best model prediction
X_opt = X_train[:, [0,3,5]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
X_opt_test = X_test[:, [0,3,5]]
y_pred_OLS = regressor_OLS.predict(X_opt_test)
