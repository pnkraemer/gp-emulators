# NAME: 'discrMeanMap.py'
#
# PURPOSE: Visualise discrepancy between conditional mean and MAP in 1D example

# DESCRIPTION: We derive the post. distribution and compute CM and MAP
#
# AUTHOR: NK, kraemer(at)ins.uni-bonn.de

from __future__ import division	
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
#plt.style.use('seaborn-colorblind')
plt.style.use('ggplot')

import sys
sys.path.insert(0,'../modules/')

from ptSetFcts import getPtsHalton
from quadrForm import compQuadQMC

np.random.seed(15051994)
np.set_printoptions(precision = 1)
plt.rcParams.update({'font.size': 26})
plt.rcParams['lines.linewidth'] = 4
plt.rcParams["figure.figsize"] = (12,9)

def parToObsOperator(pt):
	return  np.sin(0.1*pt)

trueInput = -2
noiseStdDev = 0.1
obsNoise = noiseStdDev * np.random.randn()
observation = parToObsOperator(trueInput) + obsNoise


def evaluatePotential(pt, nOrmConst = 1., data = observation):
	return np.exp(-1./(2.*noiseStdDev) * (data - parToObsOperator(pt))**2) / nOrmConst

def gaussMixMdlSym(pt, mean = 2, variance = 0.5):
	leftModel =  np.exp(-(pt+mean)**2/(2*variance))/(np.sqrt(2*np.pi*variance))
	rightModel =  np.exp(-(pt-mean)**2/(2*variance))/(np.sqrt(2*np.pi*variance))
	return leftModel + rightModel

def evaluatePotentialGaussianPrior(pt, nOrmConst = 1., data = observation):
	return evaluatePotential(pt) * gaussMixMdlSym(pt)


numPtsQmc = 10000
ptSet = 20 * getPtsHalton(numPtsQmc, 1) - 10
normConst = compQuadQMC(evaluatePotentialGaussianPrior, ptSet)


plotPts = np.linspace(-10,10,500)
likelihood = evaluatePotential(plotPts, normConst)
priorVals = gaussMixMdlSym(plotPts)
posteriorVals = priorVals * likelihood


def postDens(pt):
	return evaluatePotential(pt, normConst) * gaussMixMdlSym(pt)

def postDensCM(pt):
	return pt * evaluatePotential(pt, normConst) * gaussMixMdlSym(pt)

condMean = compQuadQMC(postDensCM, ptSet)
mapPts = np.linspace(-10,10,10000)
mapVals = postDens(mapPts)
MAP = mapPts[np.argmax(mapVals)]





plt.figure()
plt.xlabel("Location")
plt.ylabel("Probability")
plt.plot(plotPts, posteriorVals, label = "Posterior density $\pi^y$")
plt.plot(plotPts,  priorVals, label = "Prior density $\pi_0$")
plt.vlines(trueInput, 0, gaussMixMdlSym(trueInput) * evaluatePotential(trueInput, normConst), label = "True input value")
plt.legend()
plt.savefig("figures/discrMeanMap_priorPostDens.png", bbox_inches ="tight")
plt.show()


plt.figure()
plt.xlabel("Location")
plt.ylabel("Probability")
plt.plot(plotPts, posteriorVals, label = "Posterior density $\pi^y$")
plt.vlines(condMean, 0, gaussMixMdlSym(condMean) * evaluatePotential(condMean, normConst), linestyle = "dashed", label = "Conditional mean")
plt.vlines(MAP, 0, gaussMixMdlSym(MAP) * evaluatePotential(MAP, normConst),  linestyle = "dotted", label = "MAP")
plt.legend()
plt.savefig("figures/discrMeanMap_cmAndMap", bbox_inches ="tight")
plt.show()







