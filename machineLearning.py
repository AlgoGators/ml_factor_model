import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from sklearn import tree

data = datasets.load_diabetes()

df = pd.DataFrame(data=data.data, columns=data.feature_names)

# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Use only BMI
diabetes_X_single = diabetes_X[:, np.newaxis, 2]

index = list(range(1, 10))

n_test = 150
# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-n_test]
diabetes_X_test = diabetes_X[-n_test:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-n_test]
diabetes_y_test = diabetes_y[-n_test:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Create linear regression object
clf = linear_model.Ridge(alpha=0.1)

# Train the model using the training sets
clf.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = clf.predict(diabetes_X_test)

clf = Ridge()
coefs = []
r2s = []

alphas = np.logspace(-6, 6, 200)

for a in alphas:
    clf.set_params(alpha=a)
    clf.fit(diabetes_X_train, diabetes_y_train)
    coefs.append(clf.coef_)
    r2s.append(r2_score(diabetes_y_test, clf.predict(diabetes_X_test)))


# Create linear regression object
clf = linear_model.Ridge(alpha=clf.alpha)

# Train the model using the training sets
clf.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = clf.predict(diabetes_X_test)
# The coefficients
print("Coefficients: \n", clf.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# The coefficient of determination: 1 is perfect prediction
print("R2: %.5f" % r2_score(diabetes_y_test, diabetes_y_pred))
print(f'Intercept:{clf.intercept_}, Slope:{clf.coef_}')

# Display results
plt.figure(figsize=(20, 6))

plt.subplot(121)
ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale("log")
plt.xlabel("alpha")
plt.ylabel("weights")
plt.title("Ridge coefficients as a function of the regularization")
plt.axis("tight")

plt.subplot(122)
ax = plt.gca()
ax.plot(alphas, r2s)
ax.set_xscale("log")
plt.xlabel("alpha")
plt.ylabel("R2")
plt.title("R2 as a function of the regularization")
plt.axis("tight")

plt.show()


