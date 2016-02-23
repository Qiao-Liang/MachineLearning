import os
import os.path
import ConfigParser
import time
import numpy

objCP = ConfigParser.ConfigParser()
objCP.read("Similarity_Config.cfg")

def CountOccurrence(arrKey):
	dicKey = {}

	for key in arrKey:
		if key in dicKey:
			dicKey[key] += 1
		else:
			dicKey[key] = 1

	return dicKey

arrKeys = []   # Track all the keywords
arrDocs = []   # Track the document names
arrMtrx = []   # A 2-D array as the scratch of the final matrix

# Load the documents
print "Reading the documents..."
for root, dirs, names in os.walk(objCP.get("Data", "DocPath")):
	for nm in names:
		# Reset the temp variables
		dot = 0
		cos = 0
		docMod = 0
		arrOccs = []   # Track the occurence of each keyword in a document

		arrDocs.append(nm)

		with open(os.path.join(root, nm)) as objDoc:
			dicDocKeys = CountOccurrence(objDoc.read().splitlines())

			# Register the document keys in the keyword tracker
			for key in dicDocKeys:
				if key not in arrKeys:
					arrKeys.append(key)

			for key in arrKeys:
				if key in dicDocKeys:
					arrOccs.append(dicDocKeys[key])
				else:
					arrOccs.append(0)

			arrMtrx.append(arrOccs)

# Create the big matrix
print "Creating the big matrix..."
colCnt = len(arrKeys)   # Number of columns (the count of keywords)
rowCnt = len(arrDocs)   # Number of rows (the count of documents)
mtxFin = numpy.zeros((rowCnt, colCnt))   # Create an matrix with all 0 elements
arrTemp = []   # A local variable that temporarily holds the keywords of each document during the looping

for rw in range(0, rowCnt):
	arrTemp = arrMtrx[rw]

	for cl in range(0, colCnt):
		if cl < len(arrTemp):
			mtxFin[rw, cl] = arrTemp[cl]

mtxFin.transpose()   # The matrix is created with row for documents and columns for keywords, so we should transpose it.

# Do SVD
print "Doing SVD...(It takes time, please wait for a while)"
t = time.time()
U, Sigma, V = numpy.linalg.svd(mtxFin, full_matrices=True)
print "The SVD operation takes %f second(s)." % (time.time()-t)

print "Below is the Sigma matrix."
print Sigma