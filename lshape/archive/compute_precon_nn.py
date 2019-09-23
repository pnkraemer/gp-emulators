"""
All rights reserved, Nicholas Krämer, September 2019
"""

"""
run with:

python3 compute_precon.py <num_neighb> <string_path_to_mesh> <string_desired_name_of_output_file>

where <dim> is an integer (the spatial dimension of the mesh) and the other two variables are stirngs

- num_neighb is an int, consisting of the exact number of neighbors for each point

- the mesh.txt file looks like:
	
	1.124	1.12124
	2.1251	1.56
	...

for a 2d mesh and like

	1.124	1.12124 5.124
	2.1251	1.56	4.1251
	...

for a 3d mesh. The dimension of the mesh is read automatically from the file.

- the desired name of the output file is not expected to contain .txt. 'outputfile' as an input string gets used as 'outputfile.txt'



returns:
	a txt file containing the preconditioner matrix for thin plate spline preconditioning

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

# Module intern variables for debugging
save_txt = True
debug_mode = True
perf_thresh = 1000

# Read 
num_neighb = int(sys.argv[1])

path_to_mesh = sys.argv[2]
path_to_output = sys.argv[3] + ".txt"

ptset = np.loadtxt(path_to_mesh)
assert(len(ptset.shape) == 2)	# check whether mesh was in proper format
num_pts = len(ptset)
dim = len(ptset.T)

assert(num_neighb > 4)

cov_fct = TpsCov.fast_mtrx
polblocksize = 1 + dim

precon_vals, precon_rowidx, precon_colidx = LocalLagrange.precon_nn(ptset, num_neighb, cov_fct, polblocksize)
mtrx = np.vstack((precon_vals, precon_rowidx, precon_colidx)).T

if save_txt == True:
	np.savetxt('test.txt', mtrx, delimiter = '\t', header = 'VALS\tROWIDX\tCOLIDX', comments = '')


print("\nSaved preconditioner successfully")

if debug_mode == True:
	print("Entering debugging mode")
	assert(num_pts < perf_thresh), "Please debug with fewer points. Aborting..."

	precon = scipy.sparse.coo_matrix((precon_vals, (precon_rowidx, precon_colidx)), shape=(num_pts + polblocksize, num_pts + polblocksize))
	P = precon.toarray()
	P[num_pts:(num_pts+3), num_pts:(num_pts+3)] = np.eye(3)
	covmat = TpsCov.fast_mtrx(ptset, ptset)
	cond_covmat = covmat.dot(P)

	print('Computing GMRES')
	count = gmresCounter()

	rhs = np.random.rand(len(cond_covmat))
	rhs[len(rhs)-polblocksize::] = 0
	x0 = 0 * rhs

	x, info = spla.gmres(cond_covmat, rhs, callback = count)

	print('\t%u iterations' % count.numIter)
	print()