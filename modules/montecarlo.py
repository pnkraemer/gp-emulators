"""
NAME: montecarlo.py

PURPOSE: Monte Carlo integrators on rectangular domains

"""
import numpy as np
from pointsets import PointSet, Random, Lattice
"""
Basic Monte Carlo class, base for 
QMC and MCMC
"""
class MonteCarlo:

    def __init__(self, num_pts,  dim):
        self.num_pts = num_pts
        self.dim = dim
        self.ptset = None


    def new_ptset(self, special_bbox = None):
        num_pts = self.num_pts
        dim = self.dim

        ptset = Random(num_pts, dim)
        if special_bbox is not None:
            ptset.bbox = special_bbox
        ptset.construct_ptset()
        self.ptset = ptset

    def compute_integral_mc(self, integrand):
        if self.ptset is None:
            print("No pointset, returning 0...")
            return 0
        num_pts = self.num_pts
        ptset = self.ptset

        value = 0
        for i in range(num_pts):
            sample = self.ptset.points[i]
            value = value + integrand(sample)
        return value / (1. * num_pts)



"""
Monte Carlo Method with added pointset rules
"""
class QuasiMonteCarlo(MonteCarlo):

    def __init__(self, num_pts, dim):
        MonteCarlo.__init__(self, num_pts, dim)






















"""
Some Testing
"""
def product(pt): 
	return np.prod(pt)

num_pts = 100000
dim = 2
mc_unitsquare = MonteCarlo(num_pts, dim)

mc_unitsquare.new_ptset()

integral = mc_unitsquare.compute_integral_mc(product)
print(integral)





