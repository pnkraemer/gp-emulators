import numpy as np
import scipy.spatial

from joblib import Parallel, delayed
import multiprocessing

class gmresCounter(object):
    def __init__(self, disp=False):
        self._disp = disp
        self.numIter = 0
    def __call__(self, rk=None):
        self.numIter += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))




class LocalLagrange:

	# depreciated
	@staticmethod
	def precon_alt(ptSet, radius, kernelMtrxFct, polBlockSize):

		tree = scipy.spatial.KDTree(ptSet)
		numPts = len(ptSet)
		numNeighb = 1.0 * radius * np.log10(numPts) * np.log10(numPts)
		numNeighb = int(np.minimum(np.floor(numNeighb), numPts))
		k2 = numNeighb + polBlockSize
		preconVals = np.zeros(k2 * numPts)
		preconRowIdx = np.zeros(k2 * numPts)
		preconColIdx = np.zeros(k2 * numPts)
		for idx in range(numPts):
			distNeighb, indNeighb = tree.query(ptSet[idx,:], k = numNeighb)
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
		distneighb, indneighb = tree.query(ptSet, k = numNeighb)
		for idx in range(numPts):
			distNeighb = distneighb[idx,:]
			indNeighb = indneighb[idx,:]
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

"""
	@staticmethod
	def localco(idx, tree, numPts, numNeighb, k2, distneighb, indneighb, ptSet, kernelMtrxFct, polBlockSize):
		retu = np.ones((k2, 3))
		distNeighb = distneighb[idx,:]
		indNeighb = indneighb[idx,:]
		locKernelMtrx = kernelMtrxFct(ptSet[indNeighb], ptSet[indNeighb])
		locRhs = np.zeros(len(locKernelMtrx))
		locRhs[0] = 1
		lu, piv = scipy.linalg.lu_factor(locKernelMtrx)
		locCoeff = scipy.linalg.lu_solve((lu, piv), locRhs)
		retu[:,0] = locCoeff #val
		retu[:numNeighb, 1] = indNeighb #row
		retu[numNeighb:, 1] = numPts + np.arange(polBlockSize) #row
		retu[:,2] = idx #col
		return retu


	@staticmethod
	def precon_parallel(ptSet, radius, kernelMtrxFct, polBlockSize):

		tree = scipy.spatial.KDTree(ptSet)
		numPts = len(ptSet)
		numNeighb = 1.0 * radius * np.log10(numPts) * np.log10(numPts)
		numNeighb = int(np.minimum(np.floor(numNeighb), numPts))
		k2 = numNeighb + polBlockSize
		preconVals = np.zeros(k2 * numPts)
		preconRowIdx = np.zeros(k2 * numPts)
		preconColIdx = np.zeros(k2 * numPts)
		distneighb, indneighb = tree.query(ptSet, k = numNeighb)


		num_cores = multiprocessing.cpu_count()
		Ret = Parallel(n_jobs=num_cores, backend = "multiprocessing")(delayed(LocalLagrange.localco)(i, tree, numPts, numNeighb, k2, distneighb, indneighb, ptSet, kernelMtrxFct, polBlockSize) for i in range(numPts))
		retu = np.vstack(Ret)
		return retu[:,0], retu[:,1], retu[:,2], k2
"""

