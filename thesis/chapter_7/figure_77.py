# NAME: 'rileyAlgTps.py'

# PURPOSE: Check the influence of the shifts 
# and accuracy onto Riley's Algorithm

# DESCRIPTION: I solve a system involving an exponential kernel matrix 
# and iteratively compute the approximations with Riley's algorithms
# (as in (2) on slide 42 on https://drna.padovauniversitypress.it/system/files/papers/Fasshauer-2008-Lecture3.pdf)

# AUTHOR: NK, kraemer(at)ins.uni-bonn.de

import numpy as np
# import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'../../modules/')
from pointsets import *
from covariances import *




num_pts = 500;
dim = 2;
riley_shift = 1e-2
riley_maxit = 500
riley_acc = 1e-10

ptset = Halton.construct_withzero(num_pts, dim)

covmtrx = GaussCov.fast_mtrx(ptset, ptset)
covmtrx_shift = covmtrx + riley_shift * np.identity(len(covmtrx))

rhs = np.zeros(len(ptset))
rhs[0] = 1

trueSol = np.linalg.solve(covmtrx,rhs)

startvec = np.linalg.solve(covmtrx_shift,rhs)
currit = np.copy(startvec)
counter = 0
current_relerror = np.linalg.norm(covmtrx.dot(currit) - rhs)/np.linalg.norm(rhs)
relError = np.array(current_relerror)
print("Gauss:")
while current_relerror >= riley_acc and counter <= riley_maxit:
	if counter%20 == 1:
		print("(", counter, ",", current_relerror, ")")
	counter = counter + 1
	currit = startvec + riley_shift * np.linalg.solve(covmtrx_shift, currit)
	current_relerror = np.linalg.norm(covmtrx.dot(currit) - rhs)/np.linalg.norm(rhs)
	relError = np.append(relError, np.array(current_relerror))
print("(", counter, ",", current_relerror, ")")
