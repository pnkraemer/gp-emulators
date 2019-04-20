"""
NAME: quadrature.py

PURPOSE: Quadrature formulas on rectangular domains
"""

import numpy as np
from pointsets import PointSet, Random, Lattice


"""
Quadrature base class: nodes and weights
"""
class Quadrature:

    def __init__(self, nodes, weights):
        self.nodes = nodes
        self.weights = weights

    def compute_integral(self, integrand):
        num_nodes = len(self.weights)
        approx = 0.
        for i in range(num_nodes):
            approx = approx + self.weights[i] * integrand(self.nodes[i])
        return approx


"""
Monte Carlo quadrature
"""
class MonteCarlo(Quadrature):

    def __init__(self, num_pts, dim):
        random_nodes = Random(num_pts, dim)
        weights = 1./num_pts * np.ones(num_pts)
        Quadrature.__init__(self, random_nodes.points, weights)

    @staticmethod
    def fast_approximate(num_pts, dim, integrand):
        nodes = Random(num_pts, dim)
        values = integrand(nodes.points)
        #print(values)
        return np.sum(values)/float(num_pts)






"""
Some Testing
TODO: make a testing class -- this 
snippet here is run at every import which is really annoying!!
"""
# def product(pt): 
# 	return np.prod(pt)

# num_pts = 1000000
# dim = 1
# mc = MonteCarlo(num_pts, dim)
# print(mc.compute_integral(product))





