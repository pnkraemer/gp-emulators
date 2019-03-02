# NAME: 'hellDistMean1d.py'
#
# PURPOSE: Visualise 1d example of GP approximation of posterior density
#
# DESCRIPTION: We compute the approximate posterior density based on 
# a Gaussian process with Matern covariance, plot the approximate and true
# density and compute the Hellinger distance of the two:
#
# AUTHOR: NK, kraemer(at)ins.uni-bonn.de

from __future__ import division	
import numpy as np
import matplotlib.pyplot as plt 
import scipy.special
from functools import partial

import sys
sys.path.insert(0,'../modules/')

from covFcts import maternCov
from kernelMtrcs import buildCovMtrx
from forwMdls import compForwFem
from ptSetFcts import getPtsLattice


np.random.seed(15051994)



def compPotentialAppr(pt, ptSet, covFct, gPCoeff, gIvenData, mEasErrVar, normalConst = 1.0):
	covMtrxLeft = buildCovMtrx(np.reshape(pt, [1, len(pt)]), ptSet, _kernel)
	predMean = covMtrxLeft.dot(gPCoeff)
	return np.exp(-np.linalg.norm(gIvenData-predMean)**2/(2*mEasErrVar))/normalConst




numPtsGP = 10

measErrVar = 0.001

mAternReg = 2.0
mAternCorrLength = 1.0
mAternScale = 1.0

mAternCov = partial(maternCov, maternReg = mAternReg, maternScale = mAternScale, maternCorrLength = mAternCorrLength)

meshWidthFem = 1.0/32.0
meshWidthFemHighRes = 1.0/1024.0
numPtsQmc = 2**(12)

observPtSet = np.linspace(0.5,0.5,1)



trueInput = np.random.rand(1)
measErr = np.random.normal(0, measErrVar, 1)
trueData = compForwFem(trueInput, observPtSet, meshWidthFemHighRes)
givenData = trueData + measErr


# Compute approximate density:
ptSetGp = getPtsHalton(numPtsGP, 1)
forwEvals = np.zeros((len(ptSetGp), 1))
for idx in range(len(forwEvals)):
	forwEvals[idx,:] = compForwFem(ptSetGp[idx,:], observPtSet, meshWidthFem)

covMtrx = buildCovMtrx(ptSetGp, ptSetGp, mAternCov)
gpCoeff = np.linalg.solve(covMtrx, forwEvals)



# Compute normalising constant Z_approx
ptSetQmc = getPtsLattice(numPtsQmc, 1)
potApprRestr = partial(compPotentialAppr, ptSet = ptSetGp, covFct = mAternCov, gPCoeff = gpCoeff, gIvenData = givenData, mEasErrVar = measErrVar, normalConst = 1.0):
normConst = compQuadQmc(potApprRestr, ptSetQmc)

































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




