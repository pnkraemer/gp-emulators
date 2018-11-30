from __future__ import division	# division of integers into decimal numbers
import numpy as np
import scipy.special
import scipy.spatial
import scipy.sparse
import scipy.sparse.linalg
from scipy import interpolate


# solves -(\alpha(x,u) u'(x,a))' = 1 
# for \alpha(x,u) = \sum_{i=1}^K e^(a_i*x)
# at coefficient a with mesh width h
# and gives output measurements at pointset vec_J \subset [0,1]
def forward_operator_fem_1d(a, h, vec_J):

	def integral_exp(u, links, rechts):
		integral = 0
		for i in range(len(u)):
			integral = integral + (np.exp(u[i]*rechts)-np.exp(u[i]*links))/u[i]
		return integral

	N = int(1.0/h) + 1		# nodes

	N_interior = N - 2		# interior nodes - the outer ones are predefined
	
	# frames for stiffness matrix
	M = np.zeros((N_interior,N_interior))
	diag = np.zeros(N_interior)
	offdiag = np.zeros(N_interior-1)

	# rhs
	rhs = np.ones(N_interior) / (N_interior+1)

	# set diagonal entries
	for i in range(N_interior):
		i1 = i/(N_interior+1)
		i2 = (i+2)/(N_interior+1)
		diag[i] = (N_interior+1)**2 *integral_exp( a,i1,i2)

	# set off-diagonal entries
	for i in range(N_interior-1):
		i3 = (i+1)/(N_interior+1)
		i4 = (i+2)/(N_interior+1)
		offdiag[i] = -(N_interior+1)**2 * integral_exp( a,i3,i4)

	# fill in stiffness matrix
	M = scipy.sparse.diags([diag, offdiag, offdiag], [0,-1,1], format = 'csc')

	# solve for coefficients and embedd in vector for zero-bdry conditions
	X = scipy.sparse.linalg.spsolve(M,rhs)
	XX = np.zeros(N)
	XX[1:(N_interior+1)] = X

	# find evaluation points
	xvalues =  np.linspace(0,1,N)
	f = interpolate.interp1d(xvalues, XX)

	return f(vec_J)



