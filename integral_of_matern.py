from __future__ import division	#division of integers into decimal
import matplotlib.pyplot as plt 
import numpy as np 
import scipy.special
import scipy.spatial
from functools import partial
import time
from functools import partial
from kernel import get_gppoints, build_kernelmatrix, maternrbf
from lattice_new import qmc_integral, get_latticepoints
print ''
print ''
def fcttoapproximate(x):
	for i in range(200):
		for j in range(200):
			t = i + j
	return 2*x

# define method dependent resolutions
N_qmc = 10000
N_gp = 100

# define problem dependent parameters
dim = 1
nu = 1.0
sigma = 1.0
rho = 1.0

# get maternkernel
maternkernel = partial(maternrbf, NU = nu, RHO = rho, SIGMA = sigma)

start1 = time.clock()
# pointset to approximate integral
P = get_latticepoints(N_qmc, dim)

# pointset to evaluate rhs on
X = get_gppoints(N_gp,dim)


# approximate kernel mean
kernelmean = np.zeros(N_gp)
for i in range(len(X)):
	evaluatematern = partial(maternkernel, y = X[i,:])
	kernelmean[i] = qmc_integral(evaluatematern, P)

M = build_kernelmatrix(X,X, maternkernel)
c = np.linalg.solve(M,kernelmean)

f = fcttoapproximate(X)
integral1 = c.dot(f)

end1 = time.clock()


start2 = time.clock()

integral2 = qmc_integral(fcttoapproximate, P)
end2 = time.clock()


start3 = time.clock()
fcttoapproximate(1)
end3 = time.clock()
print 'Kernel quadrature took \n\tt_1 =', end1 - start1, "seconds"
print 'QMC quadrature took \n\tt_2 =', end2 - start2, "seconds"
print 'A single evaluation of the function took \n\tt_3 =', end3 - start3, "seconds"

print '\nThe integrals are:'
print '\t int_kernel = ', integral1
print '\t int_qmc = ', integral2

print '\nWith errors:'
print '\t eps_kernel = ', np.linalg.norm(integral1)
print '\t eps_qmc = ', np.linalg.norm(integral2)
print ''

