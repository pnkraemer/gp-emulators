# NAME: 'discrMeanMap.py'
#
# PURPOSE: Visualise discrepancy between conditional mean and MAP in 1d example

# DESCRIPTION:
#
# AUTHOR: NK, kraemer(at)ins.uni-bonn.de

from __future__ import division	
import numpy as np
import matplotlib.pyplot as plt 

import sys
sys.path.insert(0,'../modules/')

from covFcts import maternCov
from covMtrcs import buildCovMtrx
from ptSetFcts import getPtsHalton
from quadrForm import compQuadQMC

np.random.seed(15051994)
np.set_printoptions(precision = 1)
plt.rcParams.update({'font.size': 20})
plt.rcParams.update({'legend.fontsize': 20})

def parToObsOperator(pt):
	return  np.sin(0.1*pt)

trueInput = -2
noiseStdDev = 0.1
obsNoise = noiseStdDev * np.random.randn()
observation = parToObsOperator(trueInput) + obsNoise
print 'observation =', observation

def evaluatePotential(pt, nOrmConst = 1., data = observation):
	return np.exp(-1./(2.*noiseStdDev) * (data - parToObsOperator(pt))**2) / nOrmConst

def gaussDens(pt, mean = 2, variance = 0.5):
	a =  np.exp(-(pt+mean)**2/(2*variance))/(np.sqrt(2*np.pi*variance))
	b =  np.exp(-(pt-mean)**2/(2*variance))/(np.sqrt(2*np.pi*variance))
	return a + b
def evaluatePotentialGaussianPrior(pt, nOrmConst = 1., data = observation):
	return evaluatePotential(pt) * gaussDens(pt)


# Compute normalising constant
numPts = 1000
ptSet = 20 * getPtsHalton(numPts, 1) - 10
normConst = compQuadQMC(evaluatePotentialGaussianPrior, ptSet)


print 'normConst =', normConst


plotPts = np.linspace(-10,10,500)
likelihood = evaluatePotential(plotPts, normConst)


def postDens(pt):
	return evaluatePotential(pt, normConst) * gaussDens(pt)

print "PPP =", compQuadQMC(postDens, ptSet)



plt.figure()
plt.plot(plotPts, gaussDens(plotPts) * likelihood, color = "darkblue", label = "Posterior density", linewidth = 4)
plt.plot(plotPts,  gaussDens(plotPts), color = "darkorange", label = "Prior density", linewidth = 4)
plt.plot(trueInput,  gaussDens(trueInput) * evaluatePotential(trueInput, normConst), "d", markersize = 10, color = "black", label = "True input")
plt.vlines(trueInput, -0.1, gaussDens(trueInput) * evaluatePotential(trueInput, normConst), linestyle = "dashed", color = "black", linewidth = 2)



# plt.title("")
plt.legend(shadow=True)
#plt.xlim((-1,1))
plt.ylim((-0.1,12))
plt.grid()
plt.savefig("figures/discrMeanMapDENSCOMP")
plt.show()


# Compute CM
def postDensCM(pt):
	return pt * evaluatePotential(pt, normConst) * gaussDens(pt)
condMean = compQuadQMC(postDensCM, ptSet)

# Compute MAP
mapPts = np.linspace(-10,10,10000)
mapVals = postDens(mapPts)
MAP = mapPts[np.argmax(mapVals)]

plt.figure()
plt.plot(plotPts, gaussDens(plotPts) * likelihood, color = "black", label = "Posterior density", linewidth = 4, alpha = 0.8)
plt.plot(condMean,  gaussDens(condMean) * evaluatePotential(condMean, normConst), "d", markersize = 10, color = "darkblue", label = "Conditional Mean")
plt.vlines(condMean, -0.1, gaussDens(condMean) * evaluatePotential(condMean, normConst), linestyle = "dashed", color = "darkblue", linewidth = 2)
plt.plot(MAP,  gaussDens(MAP) * evaluatePotential(MAP, normConst), "d", markersize = 10, color = "red", label = "MAP")
plt.vlines(MAP, -0.1, gaussDens(MAP) * evaluatePotential(MAP, normConst), linestyle = "dashed", color = "red", linewidth = 2)



# plt.title("")
plt.legend(shadow=True)
#plt.xlim((-1,1))
plt.ylim((-0.1,12))
plt.grid()
plt.savefig("figures/discrMeanMapESTIMATORS")
plt.show()







