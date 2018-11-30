from __future__ import division	#division of integers into decimal
import matplotlib.pyplot as plt 
import numpy as np 
import scipy.special
import scipy.spatial
from functools import partial

from functools import partial
from kernel import get_gppoints, build_kernelmatrix
# test evaluation of kernel using computed coefficients!!

np.set_printoptions(precision=1)


N = 10
dim = 2
nu = 1.0
sigma = 1.0
rho = 1.0



# get and plot points
X = get_gppoints(N, dim)
plt.plot(X[:,0], X[:,1], 'o')
plt.title("Tensor grid in 2 dimensions")
plt.grid()
plt.show()



# build kernelmatrix
M = build_kernelmatrix(X,X,maternkernel)

# check condition number of matrix
print np.linalg.cond(M)








