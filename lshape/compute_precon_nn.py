"""
NAME: 
compute_precon_nn.py

PURPOSE: 
Creates preconditioner matrix with a fixed number of neighbours 
for each kd-tree query.

ALL RIGHTS RESERVED:
Nicholas Krämer, September 2019


#########################################################################################################
INSTRUCTIONS:
run with:

	python3 compute_precon_nn.py <path_to_mesh> <path_to_output_file> <num_neighb>

where <> indicates a variable: usually a string, but <num_neighb> is an int. 
Please make sure that the paths work; the outputfile will be created in the process.

The function returns a txt file containing the preconditioner in sparse matrix format
#########################################################################################################
"""

import numpy as np
import scipy.sparse.linalg as spla
import sys

sys.path.insert(0,'../modules/')
from pointsets import *
from misc import *
from covariances import *
from misc import *
from locallagrange import *

# Internal variables for debugging
save_txt = True
debug_mode = False	# Set to true if gmres shall be computed
perf_thresh = 1000

# Read inputs
path_to_mesh = sys.argv[1]
path_to_output = sys.argv[2]
num_neighb = int(sys.argv[3])

# Compute problem variables
ptset = np.loadtxt(path_to_mesh)
assert(len(ptset.shape) == 2), "Please enter a mesh in the right format"	# check whether mesh was in proper format
num_pts = len(ptset)
dim = len(ptset.T)
assert(num_neighb > 4), "Please enter a larger number of neighbours"	# otherwise the local systems cannot be solved reasonably

# Set up TPS systems 
cov_fct = TpsCov.fast_mtrx
polblocksize = 1 + dim

# Compute preconditioner
precon_vals, precon_rowidx, precon_colidx = LocalLagrange.precon_nn(ptset, num_neighb, cov_fct, polblocksize)
mtrx = np.vstack((precon_vals, precon_rowidx, precon_colidx)).T

# Save preconditioner into file
if save_txt == True:
	np.savetxt(path_to_output, mtrx, delimiter = '\t', header = 'VALS\tROWIDX\tCOLIDX', comments = '')
	print("\nSaved preconditioner successfully")

# If desired - check whether preconditioner does his job
if debug_mode == True:
	print("Entering debugging mode")
	assert(num_pts < perf_thresh), "Please debug with fewer points. Aborting..."

	# Construct the matrix
	precon = scipy.sparse.coo_matrix((precon_vals, (precon_rowidx, precon_colidx)), shape=(num_pts + polblocksize, num_pts + polblocksize))
	P = precon.toarray()
	P[num_pts:(num_pts+3), num_pts:(num_pts+3)] = np.eye(3)
	covmat = TpsCov.fast_mtrx(ptset, ptset)
	cond_covmat = covmat.dot(P)

	# Compute GMRES
	count = gmresCounter()
	rhs = np.random.rand(len(cond_covmat))
	rhs[len(rhs)-polblocksize::] = 0
	x0 = 0 * rhs
	x, info = spla.gmres(cond_covmat, rhs, callback = count)

	# Output
	print('\t%u iteration(s) \t(Note: number should be reasonably small)' % count.numIter)
	print()