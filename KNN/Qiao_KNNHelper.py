import ConfigParser
import operator
import numpy as np

# This library is created by Qiao Liang on 6-Mar-2016
class KNNHelper:
	def __init__(self):
		# Read the configuration
		objCP = ConfigParser.ConfigParser()
		objCP.read("KNN_Config.cfg")

		self._CP = objCP
		self._K = objCP.getint("Data","K")   # The number of K

		# Load training data
		with open(objCP.get("Data","TrnStPath"), "r") as objSet:
			arrSet = []
			for ln in objSet:
				arrSet.append(map(float,ln.strip().split(",")))

		self._Set = arrSet   # Training set
		# Load training labels
		self._Label = open(objCP.get("Data","TrnLbPath")).read().splitlines()   # Training label

	# This function does the kNN classification for a test point. The default training data are loaded by the constructor.
	# Notice that both the TestSet and TrainSet should be in the type of Numpy.array.
	def Classify(self, TestSet, TrainSet = None, TrainLabel = None, k = None):
		# Set the default value
		if TrainSet is None:
			TrainSet = np.array(self._Set)
		if TrainLabel is None:
			TrainLabel = self._Label
		if k is None:
			k = self._K

		dicDist = {}   # Declare a dictionary to track the index of the nearest neighbors and the corresponding distances
		for idx,item in enumerate(TrainSet, start = 0):   
			dicDist[idx] = np.linalg.norm(TestSet - item)   # Calculate the distance between the test point and current training point, and store it with index.

		arrK = sorted(dicDist.items(), key = operator.itemgetter(1))[0:k]   # Declare an array to store the index of k nearest points
		dicCls = {}   # Declare a dictionary to count the appearence of each class in the k nearest points

		for rnk in arrK:
			cls = TrainLabel[rnk[0]]   # Get the target class label

			if cls in dicCls:   # If this class is already in the dictionary, just increase its count by 1
				dicCls[cls] += 1
			else:   # Else, just create a new entry for this class in the dictionary, and set its count as 1
				dicCls[cls] = 1

		return max(dicCls.iteritems(), key = operator.itemgetter(1))[0]   # Return the class which is the majority in the k nearest points

	# Properties
	@property
	def Config(self):
		return self._CP
	
	@property
	def K(self):
		return self._K

	@property
	def TrainingSet(self):
		return self._Set

	@property
	def TrainingLabels(self):
		return self._Label