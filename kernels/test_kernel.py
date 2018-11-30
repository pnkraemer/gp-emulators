from __future__ import division	#division of integers into decimal
import numpy as np 

from kernel import get_gppoints, build_kernelmatrix, norm_diff, maternfunction
# test evaluation of kernel using computed coefficients!!

np.set_printoptions(precision=1)


print "\nHow many locations per dimension?"
N = input("Enter:  ")

print "\nWhich dimension?"
dim = input("Enter:  ")


# Choose some hyperparameters for the Matern kernel
nu = 1.0
sigma = 1.0
rho = 1.0



# get and plot points
X = get_gppoints(N, dim)



def maternkernel(x, y, ORD = None, NU = nu, SIGMA = sigma, RHO = rho):
	r = norm_diff(x, y, ORD)
	return maternfunction(r, NU, SIGMA, RHO)

# build kernelmatrix
M = build_kernelmatrix(X,X,maternkernel)

# check condition number of matrix
print "\nCondition number of the kernel matrix:"
print "\tCond(K) =", np.linalg.cond(M)
print ""








