import ConfigParser
import time
import datetime
import math
import numpy as np
from sklearn import linear_model
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error

objCP = ConfigParser.ConfigParser()
objCP.read("Regression_Config.cfg")

fold = objCP.getint("Misc","Fold")

def BuildMatrix(arrData):
	y = []
	X = []

	# Read the data file
	arrTemp = []
	for data in arrData:
		arrTemp = data.strip().split(' ')
		y.append(arrTemp[0])   # The first colume of each row is the target y

		arrX = []
		for val in range(1, len(arrTemp)):
			arrX.append(arrTemp[val].split(':')[1])   # The rest columns are features of input X

		X.append(arrX)

	# Return X as a matrix, y as an array, whose elements are converted into float, and the number of examples as integer
	return np.matrix(X).astype(np.float),np.array(y).astype(np.float),len(y)

# Load example data
arrExp = open(objCP.get("Data","DataPath")).read().splitlines()

# Cross validations
# Load the examples
X,y,n = BuildMatrix(arrExp)
print "The number of examples is %d" % n

objKF = KFold(n, n_folds=fold, shuffle=False, random_state=None)

# Ridge regression
print "Doing Ridge cross validation..."
objRdg = linear_model.RidgeCV(cv=objKF)
objRdg.fit(X,y)

print "The optimal lamda for Ridge regression is %f" % (n*objRdg.alpha_)

# Lasso regression
print "Doing Lasso cross validation..."
objLss = linear_model.LassoCV(cv=objKF)
objLss.fit(X,y)

print "The optimal lamda for Lasso regression is %f" % (n*objLss.alpha_)