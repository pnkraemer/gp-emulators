# This file compares the Hellinger distance of the true posterior density
# and the approximate posterior density derive by 
# a Gaussian process approximation

# The underlying FEM is a 1d problem
# The covariance structure is a Matern kernel
# The integrals are approximated with QMC
# on a randomly shifted lattice
# The GP approximation is on a uniform tensor grid


from __future__ import division	# division of integers into decimal numbers
import numpy as np
import matplotlib.pyplot as plt 
import scipy.special
import time
from functools import partial

from fem.fem_1d import forward_operator_fem_1d
from quadratures.quadratures import get_latticepoints_unitsquare, qmc_integral
from kernels.kernel import get_gppoints, build_kernelmatrix, norm_diff, maternfunction

# Set seed for random number generators
np.random.seed(1)


#-------------------------------------------
# 0. Determine parameters
#-------------------------------------------

# Let the user choose some things
print "\nWhich regularity for the Matern kernel?"
dummy_nu = input("\tEnter: ")


print "\nHow many points for GP approximation per dimension?"
dummy_gp = input("\tEnter: ")

dummy_yesno = 0
#if dummy_K == 1:
#	print "\nDo you want to see a plot of the densities?"
#	dummy_yesno = input("\tEnter: ")

start0 = time.clock()


# Problem dependent parameters
nu = dummy_nu
K = 1
J = 1
measurement_error = 0.001			#sigma_eta
rho = 1.0
sigma = 1.0


# Method dependent resolutions
h_fem = 1.0/32.0
h_superfine = 1.0/1024.0
N_gp_1d = dummy_gp
N_qmc = 2**(12)

# build other parameters
if J <= 1:
	observationpoints = np.linspace(0.5,0.5,1)
else:
	observationpoints = np.linspace(0,1,J)

# simplify some functions
solvefem = partial(forward_operator_fem_1d, vec_J = observationpoints)

def maternrbf(x, y, NU, RHO, SIGMA, normorder = 2):
	r = norm_diff(x, y, normorder)
	return maternfunction(r, NU, RHO, SIGMA)

maternkernel = partial(maternrbf, NU = nu, RHO = rho, SIGMA = sigma)


print "\nComputing the solution for"
print "\tnu =", nu
print "\tK =", K
print "\tJ =", J

print "\nUsing the resolutions"
print "\th_fem =", h_fem
print "\th_superfine =", h_superfine
print "\tN_gp =", N_gp_1d
print "\tN_gp^K =", N_gp_1d**K
print "\tN_qmc =", N_qmc

#-------------------------------------------
# 1. Simulate "given" data
#-------------------------------------------

# Construct true solution and add noise
u_actual = np.random.rand(K) * 2 - 1
system_data = solvefem(u_actual, h_superfine)
noise = np.random.normal(0,measurement_error,J)
system_data = system_data + noise


#-------------------------------------------
# 2. Compute approximate density
#-------------------------------------------

# Construct data
GP_grid = get_gppoints(N_gp_1d, K)
GP_data = np.zeros((len(GP_grid), J))
for i in range(len(GP_grid)):
	GP_data[i,:] = forward_operator_fem_1d(GP_grid[i,:], h_fem, observationpoints)


# build kernelmatrix
M = build_kernelmatrix(GP_grid,GP_grid,maternkernel)

# compute coefficients
GP_coeff = np.linalg.solve(M, GP_data)


# evaluates $\mathcal{G}_N$ at a SINGLE point $P$
def evaluate_potential_approx(_P, _XXX, _kernel, _coefficients, _y, _sigmaeta):
	_PP = np.reshape(_P, [1, len(_P)])
	_M = build_kernelmatrix(_PP, _XXX, _kernel)
	_GU = _M.dot(_coefficients)
	return np.exp(-np.linalg.norm(_y-_GU)**2/(2*_sigmaeta))

# Compute normalising constant Z_approx
QMC_pointset = get_latticepoints_unitsquare(N_qmc, K)
QMC_pointset = 2 * QMC_pointset - 1

# turn into a function depending only on the point where the pom op is evaluated
largedensity = partial(evaluate_potential_approx, _XXX = GP_grid, _kernel = maternkernel, _coefficients = GP_coeff, _y = system_data, _sigmaeta = measurement_error)

# compute normalisation constant
Z_approx = qmc_integral(largedensity, QMC_pointset)

# define approximate density function
def density(x, laden, _Z):
	return laden(x)/_Z
approx_density = partial(density, laden = largedensity, _Z = Z_approx)



# plot approximate posterior density
#grideval = np.zeros(len(GP_grid))
#for i in range(len(GP_grid)):
#	grideval[i] = approx_density(GP_grid[i,:])
#plt.plot(GP_grid, grideval, '-', color = '')
#plt.plot(u_actual, approx_density(u_actual), 'o', color ='r')
#plt.grid()
#plt.title("Approximate posterior density")
#plt.show()

# Define density function




#-------------------------------------------
# 2. Compute true density
#-------------------------------------------

# define evaluation of potential
def evaluate_potential(_P, _y, _sigmaeta, _hfem):
	_GU = solvefem(_P, _hfem)
	return np.exp(-np.linalg.norm(_y-_GU)**2/(2*_sigmaeta))
potent = partial(evaluate_potential, _y = system_data, _sigmaeta = measurement_error, _hfem = h_fem)

# compute normalisation constant
Z = qmc_integral(potent, QMC_pointset)

# define density function
fulldensity = partial(density, laden = potent, _Z = Z)

# plot FEM density
if dummy_yesno==1:
	grid = np.linspace(-1,1,250)
	grid = np.reshape(grid, [250,1])
	grideval = np.zeros(len(grid))
	grideval2 = np.zeros(len(grid))
	for i in range(len(grid)):
		grideval[i] = fulldensity(grid[i])
		grideval2[i] = approx_density(grid[i])
	plt.plot(grid, grideval, '-', color = 'blue', linewidth = 3)
	plt.plot(grid, grideval2, '-', color = 'red', linewidth = 3)
	plt.axvline(x=u_actual, linewidth = 1.5, color = 'black')
	plt.legend(["Posterior density", "GP approximation", "True value of input parameter"])
	plt.grid()
	plt.title("Posterior density and GP approximation")
	plt.show()


#-------------------------------------------
# 3. Calculate Hellinger distances
#-------------------------------------------

def hellingerfct(pt, approxdens, fulldens):
	a = np.sqrt(approxdens(pt))
	b = np.sqrt(fulldens(pt))
	return (a-b)**2
hellfct = partial(hellingerfct, approxdens = approx_density, fulldens = fulldensity)

dist2squared = qmc_integral(hellfct, QMC_pointset)
dist = np.sqrt(0.5*dist2squared)

print '\nThe Hellinger distance is given by'
print '\t   dist =', '{:.1e}'.format(dist)
print '\t2dist^2 =', '{:.1e}'.format(dist2squared)


print '\nOverall time:'
end0 = time.clock()
print "\tt = ", np.round(end0-start0, 2), 'seconds\n\n\n'




