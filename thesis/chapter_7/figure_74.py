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


dim = 2
num_reps = 4



print("\nGaussian:")
for i in range(num_reps):
	num_pts = 2**(i+3)
	ptset = Halton.construct_withzero(num_pts,dim)
	covmat = GaussCov.fast_mtrx(ptset,ptset)
	print("(", num_pts, ",", np.abs(np.min(np.linalg.eigvals(covmat))), ")")


print("\nMatern:")
for i in range(num_reps):
	num_pts = 2**(i+3)
	ptset = Halton.construct_withzero(num_pts,dim)
	covmat = MaternCov.fast_mtrx(ptset,ptset)
	print("(", num_pts, ",", np.abs(np.min(np.linalg.eigvals(covmat))), ")")

print("\nExponential:")
for i in range(num_reps):
	num_pts = 2**(i+3)
	ptset = Halton.construct_withzero(num_pts,dim)
	covmat = ExpCov.fast_mtrx(ptset,ptset)
	print("(", num_pts, ",", np.abs(np.min(np.linalg.eigvals(covmat))), ")")











