import numpy as np
import scipy.spatial


class LocalLagrange:

	"""
	Computes preconditioner in sparsematrix format by finding
	a fixed amount of neighbours for each node
	NOTE: use "precon_nn" instead of "precon_n" to distinguish 
	from "precon_h" below
	"""
	@staticmethod
	def precon_nn(ptSet, numNeighb, kernelMtrxFct, polBlockSize):
		tree = scipy.spatial.KDTree(ptSet)
		numPts = len(ptSet)
		numNeighb = np.minimum(numPts, numNeighb)
		k2 = numNeighb + polBlockSize
		preconVals = np.zeros(k2 * numPts)
		preconRowIdx = np.zeros(k2 * numPts)
		preconColIdx = np.zeros(k2 * numPts)
		__, indneighb = tree.query(ptSet, k = numNeighb)
		for idx in range(numPts):
			#distNeighb = distneighb[idx,:]
			indNeighb = indneighb[idx,:]
			locKernelMtrx = kernelMtrxFct(ptSet[indNeighb,:], ptSet[indNeighb,:])
			locRhs = np.zeros(len(locKernelMtrx))
			locRhs[0] = 1	# tree.query gives indices out ordered
			lu, piv = scipy.linalg.lu_factor(locKernelMtrx)
			locCoeff = scipy.linalg.lu_solve((lu, piv), locRhs)
			preconVals[idx*k2:(idx+1)*k2] = np.copy(locCoeff)
			preconRowIdx[idx*k2:(idx*k2 + numNeighb)] = np.copy(indNeighb)
			preconRowIdx[(idx*k2 + numNeighb):(idx+1)*k2] = np.copy(numPts + np.arange(polBlockSize))
			preconColIdx[idx*k2:(idx+1)*k2] = np.copy(idx * np.ones(k2))
		return preconVals, preconRowIdx, preconColIdx

	"""
	Computes preconditioner in sparsematrix format by using a fixed amount of neighbours for each node
	"""
	@staticmethod
	def precon_h(ptSet, radius, kernelMtrxFct, polBlockSize):
		tree = scipy.spatial.KDTree(ptSet)
		numPts = len(ptSet)
		preconVals = np.array([])
		preconRowIdx = np.array([])
		preconColIdx = np.array([])
		for idx in range(numPts):
			indNeighb = tree.query_ball_point(ptSet[idx,:], r = radius)
			indNeighb = np.array(indNeighb)
			locKernelMtrx = kernelMtrxFct(ptSet[indNeighb,:], ptSet[indNeighb,:])
			locRhs = np.zeros(len(locKernelMtrx))
			locRhs[np.where(indNeighb==idx)] = 1 # tree.query gives indices out unordered
			lu, piv = scipy.linalg.lu_factor(locKernelMtrx)
			locCoeff = scipy.linalg.lu_solve((lu, piv), locRhs)
			preconVals = np.append(preconVals, locCoeff)
			preconRowIdx = np.append(preconRowIdx, indNeighb)
			preconRowIdx = np.append(preconRowIdx, numPts + np.arange(polBlockSize))
			preconColIdx = np.append(preconColIdx, idx * np.ones(len(indNeighb) + polBlockSize))
		return preconVals, preconRowIdx, preconColIdx



"""
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
			locKernelMtrx = kernelMtrxFct(ptSet[indNeighb,:], ptSet[indNeighb,:])
			locRhs = np.zeros(len(locKernelMtrx))
			locRhs[0] = 1
			lu, piv = scipy.linalg.lu_factor(locKernelMtrx)
			locCoeff = scipy.linalg.lu_solve((lu, piv), locRhs)
			preconVals[idx*k2:(idx+1)*k2] = np.copy(locCoeff)
			preconRowIdx[idx*k2:(idx*k2 + numNeighb)] = np.copy(indNeighb)
			preconRowIdx[(idx*k2 + numNeighb):(idx+1)*k2] = np.copy(numPts + np.arange(polBlockSize))
			preconColIdx[idx*k2:(idx+1)*k2] = np.copy(idx * np.ones(k2))
		return preconVals, preconRowIdx, preconColIdx, k2
"""

