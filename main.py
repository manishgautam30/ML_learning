import matplotlib as plt
import numpy as np
from sklearn import datasets, linear_model

diabetes=datasets.load_diabetes()
print(diabetes.DESCR)
diabetes_X=diabetes.data[:,np.newaxis,2]