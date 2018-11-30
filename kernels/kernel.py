from __future__ import division	#division of integers into decimal
import numpy as np 
import scipy.special
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
	N1 = len(X)
	N2 = len(Y)
	M = np.zeros((len(X),len(Y)))
	for i in range(N1):
		for j in range(N2):
			M[i,j] = kernel(X[i,:], Y[j,:])
	return M


def norm_diff(x, y, ORD = None):
	return np.linalg.norm(x-y, ord = ORD)

# define kernelfunction: maternkernel and fix parameters
def maternfunction(r, NU, RHO, SIGMA):
	if r <= 0:
		return SIGMA**2
	else:
 		z = np.sqrt(2*NU)*r / RHO
 		a = SIGMA**2 * 2**(1-NU) / scipy.special.gamma(NU)
		return a * z**(NU) * scipy.special.kv(NU, z)


