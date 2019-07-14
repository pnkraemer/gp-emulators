# AUTHOR: NK, kraemer(at)ins.uni-bonn.de

import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt 

num_pts  = 97829
print("\nProcessing N = %u points"%num_pts)
mesh = spio.loadmat("DEMLOW/lshape_mesh_%u.mat"%num_pts)

nodes = mesh.get("node")

np.savetxt("./../../H2Lib/lshape/files/demlow_mesh/demlow_mesh_%d.txt"%num_pts, nodes, fmt='%.17e')
print("\nSaving successful")
