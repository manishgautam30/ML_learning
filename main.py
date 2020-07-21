import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()
diabetes_X = diabetes.data[:, np.newaxis, 2]

diabetes_X_train = diabetes_X[:-40]
diabetes_X_test = diabetes_X[-40:]

diabetes_Y_train = diabetes.target[:-40]
diabetes_Y_test = diabetes.target[-40:]

model = linear_model.LinearRegression()

model.fit(diabetes_X_train,diabetes_Y_train)

diabetes_Y_predict = model.predict(diabetes_X_test)

print("mean squared error is : ",mean_squared_error(diabetes_Y_test, diabetes_Y_predict))

print('weights',model.coef_)
print('intercept', model.intercept_)

plt.scatter(diabetes_X_test,diabetes_Y_test)
plt.show()
