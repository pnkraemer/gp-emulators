"""
NAME: quadrature.py

PURPOSE: Quadrature formulas on rectangular domains
"""

import numpy as np
from pointsets import Random, Lattice


"""
Quadrature base class: nodes and weights
"""
class Quadrature:

    """
    nodes: an (n, d) array
    weights: an (n, ) array
    """
    def __init__(self, nodes, weights):
        self.nodes = nodes
        self.weights = weights

    """
    integrand is a function that takes (n,d) points as an input
    and gives (n,) evaluations, i.e which can be vectorised 
    """
    def compute(self, integrand):
        values = integrand(self.nodes)
        return self.weights.dot(values)


    @staticmethod
    def compute_integral(integrand, nodes, weights):
        values = integrand(nodes)
        return weights.dot(values)

"""
Monte Carlo quadrature
"""
class MonteCarlo(Quadrature):

    def __init__(self, num_pts, dim):
        random_nodes = Random.construct(num_pts, dim)
        weights = np.ones(num_pts)/(1.0*num_pts) 
        Quadrature.__init__(self, random_nodes, weights)

    @staticmethod
    def compute_integral(integrand, num_pts, dim):
        nodes = Random.construct(num_pts, dim)
        #nodes = np.random.rand(num_pts, dim)
        values = integrand(nodes)
        return np.sum(values)/(1.0 *num_pts)


"""
Monte Carlo quadrature
"""
class QuasiMonteCarlo(Quadrature):

    def __init__(self, num_pts, dim):
        nodes = Lattice.construct(num_pts, dim)
        weights = np.ones(num_pts)/(1.0*num_pts) 
        Quadrature.__init__(self, random_nodes, weights)

    @staticmethod
    def compute_integral(integrand, num_pts, dim):
        nodes = Lattice.construct(num_pts, dim, rand_shift = True)
        #nodes = np.random.rand(num_pts, dim)
        values = integrand(nodes)
        return np.sum(values)/(1.0 *num_pts)






