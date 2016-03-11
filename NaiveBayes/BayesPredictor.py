import operator
import math
import time
import numpy
from Qiao_BayesTrainer import BayesTrainer   # This library is created by myself, Qiao Liang

# Load test data
print "Training..."
t = time.time()
objBys = BayesTrainer()   # Initialize a KNN helper object from my own library
dicWjCk, dicCk, lenSet = objBys.Train()   # Get the trained data

print "Loading test data..."
arrData = open(objBys.Config.get("Test","DataPath")).read().splitlines()
arrLabel = open(objBys.Config.get("Test","LabelPath")).read().splitlines()
arrLbOpt = numpy.unique(arrLabel)   # Get the distinct list of all labels

print "Predicting..."
dicDoc = {}   # A dictionary for documents

for data in arrData:
	arrDWO = data.split(" ")   # The array of document ID, word ID and word occurence

	docID = int(arrDWO[0])
	wrdID = int(arrDWO[1])
	wrdOcc = int(arrDWO[2])   # Word occurrence

	if docID not in dicDoc:
		dicDoc[docID] = {}

	if wrdID in dicDoc[docID]:
		dicDoc[docID][wrdID] += wrdOcc
	else:
		dicDoc[docID][wrdID] = wrdOcc

dicCls = {}   # Declare a dictionary to store the predicted label for each document
ldlt = objBys.Delta/lenSet
for docID,dic in dicDoc.iteritems():
	dicRank = {}   # Declare a new dictionary for ranking for every document

	# Iterate all the labels for the current document, and find the one that maximizes f(xk) outside this for loop
	for lb in arrLbOpt:
		dicRank[lb] = math.log(dicCk[int(lb)])   # Initialize the ranking dictionary
		for wrdID,wrdOcc in dic.iteritems():
			if wrdID in dicWjCk[int(lb)]:
				dicRank[lb] += wrdOcc*math.log(dicWjCk[int(lb)][wrdID])
			else:
				dicRank[lb] += wrdOcc*math.log(ldlt)
	
	dicCls[docID] = max(dicRank.iteritems(), key = operator.itemgetter(1))[0]   # Get the k that maximizes the f(xk)

# Compare the predicted labels with the target ones
match = 0
for key,val in dicCls.iteritems():
	if val == arrLabel[key-1]:
		match += 1

print "The accuracy is ",match*100/len(arrLabel),"%"
print "Time elapsed is %s" % str(time.time() - t)