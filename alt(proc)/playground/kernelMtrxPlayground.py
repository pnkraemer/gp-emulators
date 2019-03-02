# FILENAME: kernelMtrxPlayground.py
#
# PURPOSE: play around with a kernelmatrix class 
#
# DESCRIPTION: create a kernelmatrix class and some associated functions
#
# LAST RUN WITH: Python 3.7.2 
#
# AUTHOR: NK

import numpy as np

class kernelMtrx:

	def __init__(self, entries):
		self.entries = entries
		self.numCols = len(entries)
		self.numRows = len(entries.T)
		self.isSubMtrx = False
		
	def createSubMtrx(self, rowIdxFrom, rowIdxTo, colIdxFrom, colIdxTo):
		subMtrx = kernelMtrx(self.entries[rowIdxFrom:rowIdxTo, colIdxFrom:colIdxTo])
		subMtrx.isSubMtrx = True
		return subMtrx

	def makeZeros(numRows, numCols):
		return kernelMtrx(np.zeros((numRows, numCols)))

	def makeIdentity(numRows, numCols):
		mtrx = kernelMtrx.makeZeros(numRows, numCols)
		minDim = min(numRows, numCols)
		mtrx.entries[0:minDim, 0:minDim] = np.identity(minDim)
		return mtrx


zeroMtrx = kernelMtrx.makeIdentity(7,7)
print(zeroMtrx.entries)



