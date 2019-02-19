# FILENAME: burnInInitState.py
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
plt.rcParams['lines.markersize'] = 10
plt.rcParams["figure.figsize"] = (12,9)

np.random.seed(15051994)


def gaussDens(pt, mean = 0, variance = 1.):
	return np.exp(-(pt-mean)**2/(2*variance))/(np.sqrt(2*np.pi*variance))

propWidth = 1.
numPlots = 4
numSamp = 200
samplesSmall = np.zeros(numSamp)
currSamp = 0.5
i = 0
samplesSmall[i] = currSamp

i = i + 1



while i < numSamp:
	proposal =  currSamp + propWidth * np.random.randn()
	accProb = gaussDens(proposal)/gaussDens(currSamp)
	ratio = np.random.rand()
	if accProb < ratio:
		samplesSmall[i] = currSamp
	else:
		samplesSmall[i] = proposal
		currSamp = proposal
	i = i + 1




propWidth = 1.
numSamp = 200
samplesBig = np.zeros(numSamp)
currSamp = 12.0
i = 0
samplesBig[i] = currSamp

i = i + 1



while i < numSamp:
	proposal =  currSamp + propWidth * np.random.randn()
	accProb = gaussDens(proposal)/gaussDens(currSamp)
	ratio = np.random.rand()
	if accProb < ratio:
		samplesBig[i] = currSamp
	else:
		samplesBig[i] = proposal
		currSamp = proposal
	i = i + 1




plt.figure()
plt.plot(samplesSmall, 'X', label ="Starting at $x_0 = 0.5$")
plt.plot(samplesBig,  'X', label ="Starting at $x_0 = 12.0$")
plt.grid()
xl = plt.xlabel("Iteration")
yl = plt.ylabel("Samples")
plt.grid()
plt.legend()
plt.savefig("figures/burnInInitState", bbox_extra_artists= (xl, ), bbox_inches ="tight")
plt.show()














# plt.plot(samples, linewidth = 2)
# plt.title("Trace of Metropolis sampling")
# plt.xlabel("sample")
# plt.ylabel("mean")
# plt.show()




