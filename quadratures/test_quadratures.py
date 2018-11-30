from __future__ import division	#division of integers into decimal
import matplotlib.pyplot as plt 
import numpy as np 
from functools import partial

from lattice_new import qmc_integral, get_latticepoints


def F(x, a, b):
	return a*np.prod(x) + b*np.sum(x)




print "\nHow many QMC-points?"
N = input("Enter:  ")

print "\nWhich dimension?"
dim = input("Enter:  ")

P = get_latticepoints(N, dim)


F_reduced = partial(F, a=1, b=2)
integral = qmc_integral(F_reduced,P)


print "\nqmc-sol =", integral
print ""
