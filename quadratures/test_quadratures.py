from __future__ import division	# division of integers into decimal

import numpy as np 
from quadratures import qmc_integral, get_latticepoints_unitsquare


# define a testfunction
def F(x, a = 1, b = 2):
	return a*np.prod(x) + b*np.sum(x)




print "\nHow many QMC-points?"
N = input("Enter:  ")


print "\nWhich dimension?"
dim = input("Enter:  ")


P = get_latticepoints_unitsquare(N, dim)
P = 2 * P - 1


integral = qmc_integral(F,P)


print "\nqmc-sol =", '{:.2e}'.format(integral)
print ""
