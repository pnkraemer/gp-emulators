# NAME: 'forwMdls.py'
#
# PURPOSE: Collect different forward models
#
# DESCRIPTION: see PURPOSE
#
# AUTHOR: NK, kraemer(at)ins.uni-bonn.de

from __future__ import division
from scipy import interpolate
import scipy.sparse

# solves -(\alpha(x,u) u'(x,a))' = 1 
# for \alpha(x,u) = \sum_{i=1}^K e^(a_i*x)
# at coefficient a with mesh width h
# and gives output measurements at pointset vec_J \subset [0,1]
def compForwFem(inputVec, obsPtSet, meshwidth = 1.0/32.0):

	def compIntExp(coeff, lowerBd, upperBd):
		intSum = 0
		for idx in range(len(coeff)):
			intSum = intSum + (np.exp(coeff[idx]*upperBd)-np.exp(coeff[idx]*lowerBd))/coeff[idx]
		return intSum

	numNodes = int(1.0/meshwidth) + 1
	numIntNodes = numNodes - 2	

	stiffMtrxDiag = np.zeros(numIntNodes)
	stiffMtrxOffDiag = np.zeros(numIntNodes - 1.0)
	for idx in range(numIntNodes):
		lowBd = idx/(numIntNodes + 1.0)
		upBd = (idx + 2.0)/(numIntNodes + 1.0)
		stiffMtrxDiag[idx] = (numIntNodes + 1.0)**2 *compIntExp(inputVec,lowBd,upBd)
	for idx in range(numIntNodes-1):
		lowBd = (idx + 1.0)/(numIntNodes + 1.0)
		upBd = (idx + 2.0)/(numIntNodes + 1.0)
		stiffMtrxOffDiag[idx] = -(numIntNodes + 1.0)**2 * compIntExp(inputVec, lowBd, upBd)
	stiffMtrx = scipy.sparse.diags([stiffMtrxDiag, stiffMtrxOffDiag, stiffMtrxOffDiag], [0,-1,1], format = 'csc')

	rhs = np.ones(numIntNodes) / (numIntNodes + 1.0)

	solCoeff = scipy.sparse.linalg.spsolve(stiffMtrx,rhs)
	solCoeffWithBdry = np.zeros(N)
	solCoeffWithBdry[1:(numIntNodes+1)] = solCoeff

	nodes =  np.linspace(0,1,numNodes)
	solFct = interpolate.interp1d(nodes, solCoeffWithBdry)
	return solFct(vecMeas)

