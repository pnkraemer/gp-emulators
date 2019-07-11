# AUTHOR: NK, kraemer(at)ins.uni-bonn.de

import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial
import scipy.sparse.linalg as spla
import scipy.sparse
import scipy.linalg


import sys
sys.path.insert(0,'../modules/')

from pointsets import *
from misc import *
from covariances import *
from misc import *

save_txt = True

loc_radius = 12
dim = 2

num_pts = 65
ptset = np.loadtxt("/home/kraemer/Programmieren/H2Lib/lshape/files/lmesh_N%u.txt"%num_pts)



cov_fct = TPS.fast_mtrx
polblocksize = 1 + dim


precon_vals, precon_rowidx, precon_colidx, num_neighb = LocalLagrange.precon(ptset, loc_radius, cov_fct, polblocksize)

precon = scipy.sparse.coo_matrix((precon_vals, (precon_rowidx, precon_colidx)), shape=(num_pts + polblocksize, num_pts + polblocksize))

print('\nInitialising\n\tN = %d\n\tn = %d'%(num_pts, num_neighb - polblocksize))
print('Memory footprint of ptset:\n\t %.1f MB'%(ptset.nbytes/(1024**2)))
print('Memory footprint  of sparse matrix:\n\t %.1f MB'%(sys.getsizeof(precon.data)/(1024**2)))
print('Memory footprint of full matrix:\n\t %.1f MB'%((precon.toarray()).nbytes/(1024**2)))






if save_txt == True:
	print('Saving')
#	np.savetxt("precon_txt_tps_square/ptSet_Halton_N%d_n%d.txt"%(num_pts, num_neighb), ptset, fmt='%.17e')
	np.savetxt("precon_lshape_tps/precon_val_N%d_n%d.txt"%(num_pts, num_neighb - polblocksize), precon_vals, fmt='%.17e')
	np.savetxt("precon_lshape_tps/precon_row_N%d_n%d.txt"%(num_pts, num_neighb - polblocksize), precon_rowidx, fmt='%d')
	np.savetxt("precon_lshape_tps/precon_col_N%d_n%d.txt"%(num_pts, num_neighb - polblocksize), precon_colidx, fmt='%d')
	print('\tcomplete')
print()


P = precon.toarray()
P[num_pts:(num_pts+3), num_pts:(num_pts+3)] = np.eye(3)








covmat = TPS.fast_mtrx(ptset, ptset)
cond_covmat = covmat.dot(P)
#cond_covmat = cond_covmat[:num_pts, :num_pts]


#print("PTSET", ptset)
#print("COVMAT", covmat)
#print("PRECON", P)


print('Computing GMRES')
count = gmresCounter()






rhs = np.random.rand(len(cond_covmat))
rhs[len(rhs)-polblocksize::] = 0
x0 = 0 * rhs

x, info = spla.gmres(cond_covmat, rhs, callback = count)

#cond_covmat = cond_covmat[0:num_pts, 0:num_pts]
#print('Condition number of covariance matrix:\n\tc(M)\t= %.1f' % np.linalg.cond(covmat))

#print('Condition number of preconditioned matrix:\n\tc(MP)\t= %.1f' % np.linalg.cond(cond_covmat))

print('\t%u iterations' % count.numIter)
print()