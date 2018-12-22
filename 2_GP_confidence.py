from __future__ import division	#division of integers into decimal
import matplotlib.pyplot as plt
import numpy as np 
import scipy.special
import scipy.spatial
# test evaluation of kernel using computed coefficients!!

np.set_printoptions(precision=1)
np.random.seed(15051994)

def Gaussiancov(x, y, lam = 1.0):
	return np.exp(-1.0/(2.0*lam**2) * np.linalg.norm(x-y)**2)
def materncov(x, y, lam = 1.0):
	return np.exp(-np.sqrt(5.0)*np.linalg.norm(x-y))*(1.0 + np.sqrt(5.0)*np.linalg.norm(x-y) + 5.0/3.0 * np.linalg.norm(x-y)**2)

numPts = 3
ptSet = np.linspace(0,2,numPts)
kernMtrx = np.zeros((len(ptSet), len(ptSet)))
for idx in range(len(ptSet)):
	for jdx in range(len(ptSet)):
		kernMtrx[idx,jdx] = materncov(ptSet[idx], ptSet[jdx])

noiseVar = 0.01
kernMtrx = kernMtrx + noiseVar * np.identity(len(ptSet))
data = 4*np.random.rand(len(ptSet))-2




invKernMtrx = np.linalg.inv(kernMtrx)

numEvalPts = 150
evalPts = np.linspace(0,2,numEvalPts)
evalKernMtrxLeft = np.zeros((len(evalPts), len(ptSet)))
for idx in range(len(evalPts)):
	for jdx in range(len(ptSet)):
		evalKernMtrxLeft[idx,jdx] = materncov(evalPts[idx], ptSet[jdx])

evalKernMtrxRight = np.zeros((len(ptSet), len(evalPts)))
for idx in range(len(ptSet)):
	for jdx in range(len(evalPts)):
		evalKernMtrxRight[idx,jdx] = materncov(ptSet[idx], evalPts[jdx])

evalKernMtrxOld = np.zeros((len(evalPts), len(evalPts)))
for idx in range(len(evalPts)):
	for jdx in range(len(evalPts)):
		evalKernMtrxOld[idx,jdx] = materncov(evalPts[idx], evalPts[jdx])


predMean = evalKernMtrxLeft.dot(invKernMtrx).dot(data)
predCov = evalKernMtrxOld - evalKernMtrxLeft.dot(invKernMtrx).dot(evalKernMtrxRight)






plt.figure()
numPlots = 50
for idx in range(numPlots):
	Z1 = np.random.multivariate_normal(np.ones(len(evalPts)) * np.mean(data), evalKernMtrxOld)
	plt.plot(evalPts, Z1, linewidth = 2, alpha = 0.5)
plt.plot(evalPts, np.ones(len(evalPts)) * np.mean(data), linewidth = 4, color = "black")
plt.plot(ptSet, data, '^', markersize = 12, markerfacecolor = "black",markeredgecolor = "gray")
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
#plt.gca().spines['bottom'].set_visible(False)
#plt.gca().spines['left'].set_visible(False)
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')	#plt.title("Confidence intervals")
plt.xlim((-0.03,2.03))
plt.ylim((-3,4))
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.savefig("priorPaths")
plt.show()




plt.figure()
numPlots = 50
for idx in range(numPlots):
	Z1 = np.random.multivariate_normal(predMean, predCov)
	plt.plot(evalPts, Z1, linewidth = 2, alpha = 0.5)
plt.plot(evalPts, predMean, linewidth = 4, color = "black")
plt.plot(ptSet, data, '^', markersize = 12, markerfacecolor = "black",markeredgecolor = "gray")
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
#plt.gca().spines['bottom'].set_visible(False)
#plt.gca().spines['left'].set_visible(False)
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')	#plt.title("Confidence intervals")
plt.xlim((-0.03,2.03))
plt.ylim((-3,4))
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.savefig("condPaths")
plt.show()




plt.figure()
plt.plot(ptSet, data, '^', markersize = 12, markerfacecolor = "black",markeredgecolor = "gray")
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
#plt.gca().spines['bottom'].set_visible(False)
#plt.gca().spines['left'].set_visible(False)
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')	#plt.title("Confidence intervals")
plt.xlim((-0.03,2.03))
plt.ylim((-3,4))
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.savefig("justData")
plt.show()




