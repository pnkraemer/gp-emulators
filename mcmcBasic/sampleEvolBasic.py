# FILENAME: sampleEvolBasic.py
# PURPOSE: playground for basic mcmc 
# DESCRIPTION: set up a toy example for mcmc sampling and approximate with Metropolis sampler
# AUTHOR: NK

import matplotlib
import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import norm
np.set_printoptions(precision = 1)
plt.rcParams.update({'font.size': 20})
plt.rcParams.update({'legend.fontsize': 20})
plt.rcParams["figure.figsize"] = (16,12)

np.random.seed(15051994)


def gaussDens(pt, mean = 0, variance = 1.):
	return np.exp(-(pt-mean)**2/(2*variance))/(np.sqrt(2*np.pi*variance))

propWidth = 5.
numPlots = 4
numSamp = 10000
samples = np.zeros(numSamp)
currSamp = 1.0
i = 0
samples[i] = currSamp

i = i + 1

plotPts = np.linspace(-6, 6, 1000)
plotVals = gaussDens(plotPts)

plt.subplots(2,2)
while i < numPlots + 1:
	pltIdx = 220 + i
	plt.subplot(pltIdx)
	proposal =  currSamp + propWidth * np.random.randn()
	plt.plot(plotPts, plotVals, linewidth = 3, color = "black", label="Gaussian density")
	plt.vlines(currSamp, 0, gaussDens(currSamp), color = "darkblue", linewidth = 4, label = "Last sample")
	plt.vlines(proposal, 0, gaussDens(proposal), linestyle = "dashed", color = "red", linewidth = 4, label ="Proposal")
	accProb = gaussDens(proposal)/gaussDens(currSamp)
	ratio = np.random.rand()
	if accProb < ratio:
		samples[i] = currSamp
		plt.title("%i-Rejected"%(i))
	else:
		samples[i] = proposal
		currSamp = proposal
		plt.title("%i-Accepted"%(i))
	i = i + 1
	plt.grid()

lgd = plt.legend(loc=(-0.5, 2.3), borderaxespad=0.)
plt.savefig("figures/mcmcSamples", bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.show()





while i < numSamp:
	proposal =  currSamp + propWidth * np.random.randn()
	accProb = gaussDens(proposal)/gaussDens(currSamp)
	ratio = np.random.rand()
	if accProb < ratio:
		samples[i] = currSamp
	else:
		samples[i] = proposal
		currSamp = proposal
	i = i + 1









plt.rcParams["figure.figsize"] = (8,6)







plotPts = np.linspace(-6, 6, 1000)
plotVals = gaussDens(plotPts)
plt.figure()
plt.plot(plotPts, plotVals, linewidth = 3, color = "black", label ="Gaussian density")
plt.grid()
plt.hist(samples, bins = 50, normed = 1, color = "darkgreen", alpha = 0.5, label ="MCMC samples")
plt.legend()
plt.ylim((0,0.6))
plt.savefig("figures/histMCMC")
plt.show()














# plt.plot(samples, linewidth = 2)
# plt.title("Trace of Metropolis sampling")
# plt.xlabel("sample")
# plt.ylabel("mean")
# plt.show()




