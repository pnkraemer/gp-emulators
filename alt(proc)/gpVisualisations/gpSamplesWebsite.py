# FILENAME: gpSamplesWebsite.py
# PURPOSE: draw paths for header background on website 
# DESCRIPTION: Construct an exponential covariance matrix 
# on equidistant points, mean zero and sample from these
# AUTHOR: NK

import matplotlib
import matplotlib.pyplot as plt
import numpy as np 


np.set_printoptions(precision = 1)
np.random.seed(15051994)

plt.style.use('ggplot')
plt.rcParams["figure.figsize"] = (32,18)	# large 16:9 ratio
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["axes.facecolor"] = "#000025"






















def expKernel(pt1, pt2):
	return np.exp(-np.linalg.norm(pt1-pt2))

numPts = 60
numPlots = 20

ptSet = np.linspace(0, 1, numPts)

covMtrx = np.zeros((numPts, numPts))

for idx in range(numPts):
	for jdx in range(numPts):
		covMtrx[idx, jdx] = expKernel(ptSet[idx], ptSet[jdx])

meanVec = np.zeros(numPts)



plt.figure()
for idx in range(numPlots):
	samples = np.random.multivariate_normal(meanVec, covMtrx)
 	plt.plot(ptSet, samples, '-')
 	plt.xlim((0,1))
 	plt.gca().spines['top'].set_visible(False)
 	plt.gca().spines['right'].set_visible(False)
 	plt.gca().spines['bottom'].set_visible(False)
 	plt.gca().spines['left'].set_visible(False)
 	plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')	#plt.title("Confidence intervals")
plt.grid(False)
plt.savefig("figures/superPlt", bbox_inches ="tight")











