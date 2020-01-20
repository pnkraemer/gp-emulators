"""
All rights reserved, Nicholas Krämer, 2019
"""

import numpy as np
import scipy.sparse.linalg as spla

from gpemu.pointsets import *
from gpemu.misc import *
from gpemu.covariances import *
from gpemu.locallagrange import *


dim = 2
num_pts = 150
loc_radius = 100.0
num_neighb = loc_radius * np.log10(num_pts) * np.log10(num_pts)
num_neighb = int(np.minimum(np.floor(num_neighb), num_pts))
radius = 100.0

ptset = Lattice.construct(num_pts, dim, rand_shift = False)
cov_fct = TpsCov.fast_mtrx
polblocksize = 1 + dim

precon_vals_h, precon_rowidx_h, precon_colidx_h = LocalLagrange.precon_h(ptset, radius, cov_fct, polblocksize)
precon_vals_nn, precon_rowidx_nn, precon_colidx_nn = LocalLagrange.precon_nn(ptset, num_neighb, cov_fct, polblocksize)

precon_h = scipy.sparse.coo_matrix((precon_vals_h, (precon_rowidx_h, precon_colidx_h)), shape=(num_pts + polblocksize, num_pts + polblocksize))
precon_nn = scipy.sparse.coo_matrix((precon_vals_nn, (precon_rowidx_nn, precon_colidx_nn)), shape=(num_pts + polblocksize, num_pts + polblocksize))


discr = np.linalg.norm(precon_h.toarray() - precon_nn.toarray(), ord=1) / (num_pts**2) # discrepancy
print("Avg. error per element: %.1e (~1e-15?)"%discr)








































