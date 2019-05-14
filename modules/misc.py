import numpy as np
import scipy.spatial

class gmresCounter(object):
    def __init__(self, disp=False):
        self._disp = disp
        self.numIter = 0
    def __call__(self, rk=None):
        self.numIter += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))




class LocalLagrange:

	@staticmethod
	def precon(ptSet, radius, kernelMtrxFct, polBlockSize):

		tree = scipy.spatial.KDTree(ptSet)
		numPts = len(ptSet)
		numNeighb = 1.0 * radius * np.log10(numPts) * np.log10(numPts)
		numNeighb = int(np.minimum(np.floor(numNeighb), numPts))
		k2 = numNeighb + polBlockSize
		preconVals = np.zeros(k2 * numPts)
		preconRowIdx = np.zeros(k2 * numPts)
		preconColIdx = np.zeros(k2 * numPts)
		for idx in range(numPts):

			distNeighb, indNeighb = tree.query(ptSet[idx], k = numNeighb)
			locKernelMtrx = kernelMtrxFct(ptSet[indNeighb], ptSet[indNeighb])
			locRhs = np.zeros(len(locKernelMtrx))
			locRhs[0] = 1
			lu, piv = scipy.linalg.lu_factor(locKernelMtrx)
			locCoeff = scipy.linalg.lu_solve((lu, piv), locRhs)
			preconVals[idx*k2:(idx+1)*k2] = locCoeff
			preconRowIdx[idx*k2:(idx*k2 + numNeighb)] = indNeighb
			preconRowIdx[(idx*k2 + numNeighb):(idx+1)*k2] = numPts + np.arange(polBlockSize)
			preconColIdx[idx*k2:(idx+1)*k2] = idx
			
		return preconVals, preconRowIdx, preconColIdx, k2

