"""
NAME: montecarlo.py

PURPOSE: Monte Carlo integrators on rectangular domains

"""

import numpy as np
from pointsets import PointSet, Random, Lattice


"""
Base class for MC-like methods such as QMC and MCMC
Essentially, the information is stored in the pointset
"""
class MonteCarlo:

    def __init__(self, num_pts, dim):
        self.pointset = PointSet(num_pts, dim)

    def new_pointset(self, special_bbox = None):
        num_pts = self.pointset.num_pts
        dim = self.pointset.dim

        pointset = Random(num_pts, dim)
        if special_bbox is not None:
            pointset.bbox = special_bbox
        pointset.construct_pointset()
        self.pointset = pointset

    def compute_integral_mc(self, integrand):
        num_pts = self.pointset.num_pts
        points = self.pointset.points

        value = 0
        for i in range(num_pts):
            sample = points[i]
            value = value + integrand(sample)
        return value / (1. * num_pts)



"""
Quasi Monte Carlo as Monte Carlo with added pointset information
"""
class QuasiMonteCarlo(MonteCarlo):

    def __init__(self, num_pts, dim):
        MonteCarlo.__init__(self, num_pts, dim)


"""
Some Testing
TODO: make a testing class -- this 
snippet here is run at every import which is really annoying!!
"""
def product(pt): 
	return np.prod(pt)

num_pts = 100000
dim = 2
mc_unitsquare = MonteCarlo(num_pts, dim)

mc_unitsquare.new_pointset()


integral = mc_unitsquare.compute_integral_mc(product)
print(integral)





