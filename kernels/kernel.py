from __future__ import division	#division of integers into decimal
import matplotlib.pyplot as plt 
import numpy as np 
import scipy.special
from functools import partial
import scipy.spatial




# construct uniform tensor grid with N^dim points in dim dimensions
# N is number of gridpoints per dimension
def get_gppoints(N, dim):
	single_dimension = np.zeros((N, dim))
	for i in range(dim):
		single_dimension[:,i] = np.linspace(-1,1,N)
	if dim > 1:
		tensorgrid = np.meshgrid(*single_dimension.T)
		fullgrid = np.zeros((N**dim, dim))
		for i in range(dim):
			fullgrid[:,i] = np.array(tensorgrid[i]).flatten()
	elif dim <= 1:
		fullgrid = single_dimension
	return fullgrid

#param_matern = [nu, rho, sigma] (alphabetically)
def build_kernelmatrix(X, Y, kernel):
	M = np.zeros((len(X),len(Y)))
	for i in range(len(X)):
		for j in range(len(Y)):
			M[i,j] = kernel(X[i,:], Y[j,:])
	return M


# define kernelfunction: maternkernel and fix parameters
def maternrbf(x, y, NU, RHO, SIGMA):
	if np.linalg.norm(x-y) == 0:
		return SIGMA**2
	else:
 		r = np.linalg.norm(x-y)
 		z = np.sqrt(2*NU)*r / RHO
 		a = SIGMA**2 * 2**(1-NU) / scipy.special.gamma(NU)
		return a * z**(NU) * scipy.special.kv(NU, z)


