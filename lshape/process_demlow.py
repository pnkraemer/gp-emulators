# AUTHOR: NK, kraemer(at)ins.uni-bonn.de


"""
run as:

python3 process_demlow.py 129
"""
import numpy as np
import scipy.io as spio
import sys

num_pts = int(sys.argv[1])
print("\nProcessing N = %u points"%num_pts)
mesh = spio.loadmat("DEMLOW/lshape_mesh_%u.mat"%num_pts)

nodes = mesh.get("node")

np.savetxt("./mesh/mesh_N%d.txt"%num_pts, nodes, fmt='%.17e')
print("\nSaving successful")

"""
import matplotlib.pyplot as plt 
plt.plot(nodes[:,0], nodes[:,1], '.')

plt.show()
"""