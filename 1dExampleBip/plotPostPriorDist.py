# NAME: 'plotPostPriorDist.py'
#
# PURPOSE: Show simple plot of posterior and prior PDS
#
# DESCRIPTION: We derive the post. distribution
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

trueInput = 1.
noiseStdDev = 0.01
obsNoise = noiseStdDev * np.random.randn()
observation = parToObsOperator(trueInput) + obsNoise


def evaluatePotential(pt, nOrmConst = 1., data = observation):
	return np.exp(-1./(2.*noiseStdDev) * (data - parToObsOperator(pt))**2) / nOrmConst

def gaussDens(pt, mean = 0.5, variance = 1.5):
	return np.exp(-(pt-mean)**2/(2*variance))/(np.sqrt(2*np.pi*variance))

def evaluatePotentialUnifPrior(pt, nOrmConst = 1., data = observation):
	return evaluatePotential(pt) * 0.5


numPtsQmc = 10000
ptSet = 2 * getPtsHalton(numPtsQmc, 1) - 1
normConst = compQuadQMC(evaluatePotentialUnifPrior, ptSet)


plotPts = np.linspace(-7,9,500)
likelihood = evaluatePotential(plotPts, normConst)
priorVals = 0.5 * np.ones(len(plotPts))
posteriorVals = priorVals * likelihood

plt.figure()
plt.xlabel("Location")
plt.ylabel("Probability")
plt.ylim((-0.1,2.5))
plt.plot(plotPts, posteriorVals, label = "Posterior density $\pi^y$")
plt.plot(plotPts,  priorVals, label = "Prior density $\pi_0$")
plt.vlines(trueInput, 0, 0.5 * evaluatePotential(trueInput, normConst), label = "True input value")
plt.legend()
plt.savefig("figures/plotPostPriorDist.png", bbox_inches ="tight")

plt.show()




