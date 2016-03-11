import os
import numpy as np
from Qiao_KNNHelper import KNNHelper   # This library is created by myself, Qiao Liang

objKNN = KNNHelper()   # Initialize a KNN helper object from my own library

with open(objKNN.Config.get("Data","TstStPath"), "r") as objTest:
	arrTest = []
	for ln in objTest:
		arrTest.append(map(float,ln.strip().split(",")))
arrTest = np.array(arrTest)   # Convert the test set into Numpy array for easier operation

# Report the classification of test points
file = objKNN.Config.get("Report","RptPath") + "Classification(k=%d)" % objKNN.K + objKNN.Config.get("Report","FileExt")

objRpt = open(file, 'w')

for t in arrTest:
	objRpt.write(objKNN.Classify(TestSet = t) + "\n")

objRpt.close()

print "A report has been created to %s" % os.path.abspath(file)