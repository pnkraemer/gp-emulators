# FILENAME: acceptRejectFirstStates.py
# PURPOSE: playground for basic mcmc 
# DESCRIPTION: set up a toy example for mcmc sampling and approximate with Metropolis sampler
# AUTHOR: NK

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np 
from scipy.stats import norm
np.set_printoptions(precision = 1)

plt.rcParams.update({'font.size': 26})
plt.rcParams['lines.linewidth'] = 4
plt.rcParams["figure.figsize"] = (24,18)
plt.rcParams["axes.titlesize"] = "medium"

np.random.seed(15051994)


def gaussDens(pt, mean = 0, variance = 1.):
	return np.exp(-(pt-mean)**2/(2*variance))/(np.sqrt(2*np.pi*variance))

propWidth = 3
numPlots = 4
numSamp = 20000
samples = np.zeros(numSamp)
currSamp = 0.0
i = 0
samples[i] = currSamp

i = i + 1

plotPts = np.linspace(-6, 6, 1000)
plotVals = gaussDens(plotPts)

plt.subplots(2,2)
plt.subplots_adjust(wspace = 0.2, hspace = 0.3)
while i < numPlots + 1:
	pltIdx = 220 + i
	plt.subplot(pltIdx)
	plt.xlabel("Location")
	plt.ylabel("Probability")
	proposal =  currSamp + propWidth * np.random.randn()
	plt.plot(plotPts, plotVals, label="Gaussian density")
	plt.vlines(currSamp, 0, gaussDens(currSamp), label = "Last sample")
	plt.vlines(proposal, 0, gaussDens(proposal), linestyle = "dashed", label ="Proposal")
	accProb = gaussDens(proposal)/gaussDens(currSamp)
	ratio = np.random.rand()
	if accProb < ratio:
		samples[i] = currSamp
		plt.title("i = %i (Rejected)"%(i))
	else:
		samples[i] = proposal
		currSamp = proposal
		plt.title("i = %i (Accepted)"%(i))
	i = i + 1
	plt.grid(True)

lgd = plt.legend(loc=(-0.8, 2.45), borderaxespad=0., ncol = 3)
plt.savefig("figures/acceptRejectFirstStates", bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.show()




