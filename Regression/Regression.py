import ConfigParser
import time
import datetime
import math
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

objCP = ConfigParser.ConfigParser()
objCP.read("Regression_Config.cfg")

rdgLamda = objCP.getfloat("Misc","RdgLamda")   # Get the configuration of lamda for Ridge regression as float
lssLamda = objCP.getfloat("Misc","LssLamda")   # Get the configuration of lamda for Lasso regression as float
expNum = objCP.getint("Misc","ExpNum")   # Get the configuration of example number

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

# Regressions
# Load the examples
X,y,n = BuildMatrix(arrExp[0:expNum])   # n refers to the number of training examples
print "The number of examples is %d." % n

# Ridge regression
print "Doing Ridge regression..."
objRdg = linear_model.Ridge(alpha=rdgLamda/n)
objRdg.fit(X,y)

print "Below is the optimal solution for Ridge regression"
print objRdg.coef_

# Lasso regression
print "Doing Lasso regression..."
objLss = linear_model.Lasso(alpha=lssLamda/n, fit_intercept=False)
objLss.fit(X,y)

print "Below is the optimal solution for Lasso regression"
print objLss.coef_

# Predictions
if expNum >= len(arrExp):
	print "All the example data are used as training data, no more for validation!"
else:
	# Load the validations
	pX,py,m = BuildMatrix(arrExp[expNum:])

	# Calculate the root mean square error of Ridge regression
	rmsRdg = math.sqrt(mean_squared_error(py, objRdg.predict(pX)))

	print "When lamda is %f, the root mean square error of Ridge regression is %f" % (rdgLamda, rmsRdg)

	# Calculate the root mean square error of Lasso regression
	rmsLss = math.sqrt(mean_squared_error(py, objLss.predict(pX)))

	print "When lamda is %f, the root mean square error of Lasso regression is %f" % (lssLamda, rmsLss)