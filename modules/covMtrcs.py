# NAME: 'kernelMtrcs.py'
#
# PURPOSE: Collection of scripts to construct kernel matrices
#
# DESCRIPTION: see PURPOSE
#
# AUTHOR: NK, kraemer(at)ins.uni-bonn.de

import numpy as np
from miscFcts import sph00, sph10, sph11, sph12, sph20, sph21, sph22, sph23, sph24, sph30, sph31, sph32, sph33, sph34, sph35, sph36

np.random.seed(15051994)
np.set_printoptions(precision = 2)

def buildCovMtrxShift(ptSetOne, ptSetTwo, covFct, shiftPar):
	lenPtSetOne = len(ptSetOne)
	lenPtSetTwo = len(ptSetTwo)
	if lenPtSetOne != lenPtSetTwo:
		print "The pointsets do not align... return 0"
		return 0
	covMtrx = np.zeros((lenPtSetOne, lenPtSetTwo))
	for idx in range(lenPtSetOne):
		for jdx in range(lenPtSetTwo):
				covMtrx[idx,jdx] = covFct(ptSetOne[idx,:], ptSetTwo[jdx,:])
	return covMtrx + shiftPar * np.identity(lenPtSetOne)

def buildCovMtrx(ptSetOne, ptSetTwo, covFct):
	lenPtSetOne = len(ptSetOne)
	lenPtSetTwo = len(ptSetTwo)
	covMtrx = np.zeros((lenPtSetOne, lenPtSetTwo))
	for idx in range(lenPtSetOne):
		for jdx in range(lenPtSetTwo):
			covMtrx[idx,jdx] = covFct(ptSetOne[idx,:], ptSetTwo[jdx,:])
	return covMtrx






