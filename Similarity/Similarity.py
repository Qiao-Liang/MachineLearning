import math
import sys
import os
import os.path
import operator
import ConfigParser
import time
import datetime
 
objCP = ConfigParser.ConfigParser()
objCP.read("Similarity_Config.cfg")
topRk = objCP.getint("Misc","TopRank")

def CountOccurrence(arrKey):
	dicKey = {}

	for key in arrKey:
		if key in dicKey:
			dicKey[key] += 1
		else:
			dicKey[key] = 1

	return dicKey

def GenerateReport(rptName, arrRank):
	objRpt = open(rptName, 'w')
	objRpt.write("Document, Similarity\n")
	for val in arrRank:
		objRpt.write("%s,%s\n" % (val[0],val[1]))

	objRpt.close()

srt = time.time()   # Start the timer

# Load the queries
qryFile = sys.argv[1]
objQry = open(objCP.get("Data", "QryPath") + qryFile)
dicQryKeys = CountOccurrence(objQry.read().splitlines())

# Calculate the module of the query in question
qryMod = 0
for key in dicQryKeys:
	qryMod += pow(dicQryKeys[key],2)

# Search through the documents
dicDocKeys = {}   # Temporarily counts the occurrence of each key word in each document
dicDotRank = {}   # Keep track of the dot product rank
dicCosRank = {}   # Keep track of the cosine rank

for root, dirs, names in os.walk(objCP.get("Data", "DocPath")):
	for nm in names:
		# Reset the temp variables
		dot = 0
		cos = 0
		docMod = 0

		with open(os.path.join(root, nm)) as objDoc:
			dicDocKeys = CountOccurrence(objDoc.read().splitlines())

			for key in dicQryKeys:
				if key in dicDocKeys:
					dot += dicQryKeys[key] * dicDocKeys[key]

			dicDotRank[nm] = int(dot)   # Log the rank of the dot production  

			# Calculate the sum of square for each key word occurence in the document
			for key,val in dicDocKeys.iteritems():
				docMod += pow(val,2)

			cos = dot/(math.sqrt(qryMod*docMod))

			dicCosRank[nm] = float(cos)   # Log the rank of the

# Sort the final ranks
arrDotRank = sorted(dicDotRank.items(), key=operator.itemgetter(1), reverse=True)
arrCosRank = sorted(dicCosRank.items(), key=operator.itemgetter(1), reverse=True)

# Display the results
print "The time elapsed is %f second(s)." % (time.time() - srt)
print "Below is the top %d of dot production similarity." % topRk
print arrDotRank[0:topRk]
print "Below is the top %d of cosine similarity." % topRk
print arrCosRank[0:topRk]

# Generate the report files
rptExt = objCP.get("Report","FileExt")
rptDot = objCP.get("Report", "DotPath") + "Dot_product_report-query_%s-%s" % (qryFile, datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")) + rptExt
rptCos = objCP.get("Report", "CosPath") + "Cosine_report-query_%s-%s" % (qryFile, datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")) + rptExt

GenerateReport(rptDot, arrDotRank[0:topRk])
print "Dot production report is created at %s" % os.path.abspath(rptDot)
GenerateReport(rptCos, arrCosRank[0:topRk])
print "Cosine report is created at %s" % os.path.abspath(rptCos)