# FILENAME: sampleEvolBasic.py
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
plt.rcParams["figure.figsize"] = (12,9)

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













plotPts = np.linspace(-6, 6, 1000)
plotVals = gaussDens(plotPts)
plt.figure()
plt.xlabel("Location")
plt.ylabel("Probability / Rel. Frequency")
plt.plot(plotPts, plotVals, label ="Gaussian density")
plt.hist(samples, bins = 25, density = 1, label ="MCMC samples")
plt.legend()
plt.grid(True)
plt.ylim((0,0.5))
plt.savefig("figures/histMCMC", bbox_inches ="tight")
plt.show()














# plt.plot(samples, linewidth = 2)
# plt.title("Trace of Metropolis sampling")
# plt.xlabel("sample")
# plt.ylabel("mean")
# plt.show()




