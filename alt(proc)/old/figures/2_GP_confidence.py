from __future__ import division	#division of integers into decimal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np 
import scipy.special
import scipy.spatial
from functools import partial

from functools import partial
from kernels import get_gppoints, build_kernelmatrix, maternrbf
# test evaluation of kernel using computed coefficients!!

np.set_printoptions(precision=1)
np.random.seed(4)
plt.rcParams.update({'font.size': 22})




# plots samples from given mean and covariance at points 
# together with data in an empty plot
def plotGP_data(Minverse, data, kernelmatrix):

	# extract info from data
	approxpts = data[:,0]
	approxpts = np.reshape(approxpts, (len(approxpts), 1))
	y = data[:,1]
	y = np.reshape(y, (len(y),1))

	# determine axis limits
	xmin = np.min(approxpts)
	xmax = np.max(approxpts)
	ymin = np.min(y)
	ymax = np.max(y)
	
	newpts = np.linspace(xmin, xmax, 250)
	newpts = np.reshape(newpts, (len(newpts), 1))

	# build kernel approximation matrices on new pts
	K = build_kernelmatrix(newpts, approxpts, kernelmatrix)
	K2 = build_kernelmatrix(newpts, newpts, kernelmatrix)


	# build approximation mean on newpts
	coeff = Minverse.dot(y - np.mean(y))
	approxmean = K.dot(coeff) + np.mean(y) * np.ones((len(newpts),1))

	# build approximation covariance on newpts
	Knew = K2 - K.dot(Minv).dot(K.T)

	# construct deviations
	plusdev = approxmean[:,0] + np.sqrt(Knew.diagonal())
	minusdev = approxmean[:,0] - np.sqrt(Knew.diagonal())

	# plot mean
	fig = plt.figure()
#	plt.plot(newpts[:,0], approxmean[:,0], '-', linewidth = 3, color = 'darkblue')

	# plot data
	plt.plot(approxpts, y, 'o', markersize = 10, color = 'darkslategray')

	# plot deviation
#	plt.fill_between(newpts[:,0], minusdev, plusdev, color = 'burlywood', alpha = 0.5)
	


	plt.xlim((xmin - 0.1, xmax + 0.1))
	plt.ylim((ymin - 1,ymax + 1))

	# remove frame
	plt.gca().spines['top'].set_visible(False)
	plt.gca().spines['right'].set_visible(False)
	plt.gca().spines['bottom'].set_visible(False)
	plt.gca().spines['left'].set_visible(False)
	plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')	#plt.title("Confidence intervals")

	# save figure
	plt.savefig("data")

	return 0


N = 5
nu = 1.0
sigma = 1.0
rho = 1.0

maternkernel = partial(maternrbf, NU = nu, RHO = rho, SIGMA = sigma)



# get points
X = get_gppoints(N,1)


# create data
#y = truefct(X)
y = np.random.rand(N, 1)

# build kernelmatrix
M = build_kernelmatrix(X,X,maternkernel)

# invert kernelmatrix
Minv = np.linalg.inv(M)

# create data
data = np.concatenate((X,y), axis = 1)



#Minv = np.zeros((len(M), len(M)))
#plotGP_firstdeviation(Minv, data, maternkernel)
plotGP_data(Minv, data, maternkernel)

#plt.plot(X2, approxmean[:,0] + 2*Z, '-', linewidth = 2, color = 'gray', alpha = 0.4)
#plt.plot(X2, prediction, '-', linewidth = 1, color = 'green')
#plt.plot(X2, prediction2, '-', linewidth = 1, color = 'green')
# check condition number of matrix








