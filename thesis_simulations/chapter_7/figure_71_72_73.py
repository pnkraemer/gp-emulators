# NAME: 'rileyAlgTps.py'

# PURPOSE: Check the influence of the shifts 
# and accuracy onto Riley's Algorithm

# DESCRIPTION: I solve a system involving an exponential kernel matrix 
# and iteratively compute the approximations with Riley's algorithms
# (as in (2) on slide 42 on https://drna.padovauniversitypress.it/system/files/papers/Fasshauer-2008-Lecture3.pdf)

# AUTHOR: NK, kraemer(at)ins.uni-bonn.de

import numpy as np
import sys
sys.path.insert(0,'../../modules/')
from pointsets import *
from covariances import *

import scipy.special
np.set_printoptions(precision = 1)

dim = 2
num_reps = 6

regu = 1e-4

print("\nExponential:")
for i in range(num_reps):
	num_pts = 2**(i+3)
	ptset = Halton.construct_withzero(num_pts, dim)
	covmat = ExpCov.fast_mtrx(ptset, ptset) + regu * np.eye(len(ptset))
	print("(", num_pts, ",", np.linalg.cond(covmat), ")")

print("\nMatern(nu = 2):")
mcov = MaternCov(2)
for i in range(num_reps):
	num_pts = 2**(i+3)
	ptset = Halton.construct_withzero(num_pts, dim)
	covmat = mcov.evaluate(ptset, ptset) + regu * np.eye(len(ptset))
	print("(", num_pts, ",", np.linalg.cond(covmat), ")")

print("\nTPS:")
for i in range(num_reps):
	num_pts = 2**(i+3)
	ptset = Halton.construct_withzero(num_pts, dim)
	covmat = TPS.fast_mtrx(ptset, ptset) + regu * np.eye(len(ptset) + 3)
	print("(", num_pts, ",", np.linalg.cond(covmat), ")")

print("\nGauss:")
for i in range(num_reps):
	num_pts = 2**(i+3)
	ptset = Halton.construct_withzero(num_pts, dim)
	covmat = GaussCov.fast_mtrx(ptset, ptset) + regu * np.eye(len(ptset))
	print("(", num_pts, ",", np.linalg.cond(covmat), ")")












