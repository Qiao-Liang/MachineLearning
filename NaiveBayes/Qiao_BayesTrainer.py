import ConfigParser
import operator
import math
import time
import numpy

class BayesTrainer:
	def __init__(self):
		# Read the configuration
		objCP = ConfigParser.ConfigParser()
		objCP.read("NB_Config.cfg")

		self._CP = objCP
		self._dlt = float(objCP.get("Data","Delta"))
		self._Data = open(objCP.get("Data","DataPath")).read().splitlines()
		self._Label = open(objCP.get("Data","LabelPath")).read().splitlines()

	# Properties
	@property
	def Config(self):
	  return self._CP

	@property
	def Delta(self):
	  return self._dlt
	
	# This function does the training, and return a dictionary of p(Wj|Ck) and that of P(Ck)
	def Train(self):
		dicWjCk = {}   # A dictionary to store p(Wj|Ck)
		dicWdTl = {}   # A dictionary to track the total word occurence of each label
		arrWrd = []   # A array to track the number of distinct words
		dicCk = {}   # A dictionary to store P(Ck)
		DstLb = numpy.unique(self._Label)

		# Calculate p(Ck)
		tolLb = len(self._Label)
		for lb in DstLb:
			dicWjCk[int(lb)] = {}
			dicCk[int(lb)] = float(self._Label.count(lb))/tolLb
			dicWdTl[int(lb)] = 0   # Initialize the total word occurrence dictionary

		for data in self._Data:
			arrDWO = data.split(" ")   # The array of document ID, word ID and word occurence

			docID = int(arrDWO[0])
			wrdID = int(arrDWO[1])
			wrdOcc = int(arrDWO[2])   # Word occurrence
			clbID = int(self._Label[docID - 1])   # Get the label ID of current document. Notice the arrLabel is 0-based in the array while it's actually 1-based

			# Count the occurrence of current word in its label
			if wrdID in dicWjCk[clbID]:
				dicWjCk[clbID][wrdID] += wrdOcc
			else:
				dicWjCk[clbID][wrdID] = wrdOcc

			# Count the total number of word occurrence in current label
			dicWdTl[clbID] += wrdOcc

			# Log the words
			arrWrd.append(wrdID)

		# Turn the stored word occurrence into p(wj|Ck)
		lenSet = len(numpy.unique(arrWrd))
		for lb,dics in dicWjCk.iteritems():
			for wID,wOcc in dics.iteritems():
				dics[wID] = (1 - self._dlt)*float(wOcc)/dicWdTl[lb] + self._dlt/lenSet   #len(dicWrd[lb])

		return dicWjCk, dicCk, lenSet