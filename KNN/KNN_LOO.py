import operator
import copy
import numpy as np
from Qiao_KNNHelper import KNNHelper   # This library is created by myself, Qiao Liang

# Load test data
objKNN = KNNHelper()   # Initialize a KNN helper object from my own library

lk = objKNN.Config.getint("LOO", "LOO-K")   # Get the range of k to be validated
arrOriSet = objKNN.TrainingSet   # Get the original training set
arrOriLbl = objKNN.TrainingLabels   # Get the original training labels
lenOS = len(arrOriSet)

# Do the leave-one-out
dicErr = {}   # Declare a dictionary to track the errors for each k

for k in range(1, lk + 1):   # Iteratively check the error for k from 1 to the largest k in the configuration
	print "Validating k = %d..." % k

	dicErr[k] = 0   # Set the error count to 0 before looping
	for i in range(0, lenOS):
		# Copy the original training set and labels
		arrLOOSet = copy.copy(arrOriSet)
		arrLOOLbl = copy.copy(arrOriLbl)

		# Declare the current test set and labels, meanwhile, the test set is removed from the leave-one-out training set
		arrTestSet = arrLOOSet.pop(i)
		testLbl = arrLOOLbl.pop(i)

		# Do the classification for the one test set, respect to the leave-one-out training set
		cls = objKNN.Classify(TestSet = np.array(arrTestSet), TrainSet = np.array(arrLOOSet), TrainLabel = arrLOOLbl, k = k)
		if cls != testLbl:
			dicErr[k] += 1

dicMin = min(dicErr.iteritems(), key = operator.itemgetter(1))
print "The key-value pairs below shows the number of errors for k from 1 to %d." % lk
print dicErr

# Calculate the averaged errors
for key,val in dicErr.iteritems():
	dicErr[key] = float(val)/lenOS

print "And the key-value pairs below shows the averaged leave-one-out error for 1 from 1 to %d." % lk
print dicErr
print "After all, we have the minimum error(%d) when k = %d." % (dicMin[1]/lenOS, dicMin[0])