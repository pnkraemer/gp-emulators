from __future__ import division	# division of integers into decimal numbers
import numpy as np
import matplotlib.pyplot as plt 
import scipy.special
import scipy.spatial
import scipy.sparse
import scipy.sparse.linalg
from scipy import interpolate

from fem_new import solve_fe


# determine FEM parameters
h = 1.0/1234
a = np.array([1.0, 1.0, 2.0])
J = np.linspace(0,1,111)

# solve fem problem and evaluate at the J-points
UJ = solve_fem(a, h, J)


# plot solution
plt.plot(J, UJ, '-')
plt.grid()
plt.title("FEM Solution")
plt.show()



































	# return observations
#	if _make_mistakes == 1:
#		_nu2 = np.random.normal(0,_sigma_eta,_J)
#		print 'Mistake =', _nu2
#		
#		# compute 'measurements'
#		_y = _XX[_comp.searchsorted(_xj)] + _nu2
#		print '--> _y=', _y - _nu2, 'becomes _y=', _y
#	else: