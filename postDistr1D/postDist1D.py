# NAME: 'postDist1D.py'
#
# PURPOSE: Visualise 1d example of prior and posterior density
#
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
	return np.sin(pt)

trueInput = 0.1
noiseStdDev = 0.5
obsNoise = noiseStdDev * np.random.randn()
observation = parToObsOperator(trueInput) + obsNoise
print 'observation =', observation

def evaluatePotential(pt, nOrmConst = 1., data = observation):
	return np.exp(-1./(2.*noiseStdDev) * (data - parToObsOperator(pt))**2) / nOrmConst


# Compute normalising constant
numPts = 1000
ptSet = getPtsHalton(numPts, 1)
normConst = .5  * compQuadQMC(evaluatePotential, ptSet)


print 'normConst =', normConst


plotPts = np.linspace(-1,1,100)
likelihood = evaluatePotential(plotPts, normConst)

plt.plot(plotPts, 0.5 * likelihood, color = "darkblue", label = "Posterior density", linewidth = 4)
plt.plot(plotPts, 0.5 * np.ones(len(plotPts)), color = "darkorange", label = "Prior density", linewidth = 4)
plt.plot(trueInput, 0.5 * evaluatePotential(trueInput, normConst), "d", markersize = 10, color = "black", label = "True input")
plt.vlines(trueInput, -0.1, 0.5 * evaluatePotential(trueInput, normConst), linestyle = "dashed", color = "black", linewidth = 2)



# plt.title("")
plt.legend(shadow=True)
plt.xlim((-1,1))
plt.ylim((-0.1,2))
plt.grid()
plt.savefig("figures/postDist1d.png")
plt.show()





