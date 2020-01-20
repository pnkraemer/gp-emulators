"""
NAME:
quadrature.py

PURPOSE:
Quadrature formulas on unit cubes.
"""

import numpy as np
from gpemu.pointsets import Random, Lattice


class Quadrature:
    """
    Quadrature base class.
    A quadrature rule consists of some nodes
    and some weights

    Attributes
    ----------
    nodes: (n, d) ndarray
        Quadrature nodes.
    weights: (n, ) ndarray
        Quadrature weights.
    """
    def __init__(self, nodes, weights):
        """
        Initialise quadrature rule with
        nodes and weights.

        Parameters
        ----------
        nodes: (n, d) ndarray
            Quadrature nodes.
        weights: (n, ) ndarray
            Quadrature weights.
        """
        self.nodes = nodes
        self.weights = weights

    def compute(self, integrand):
        """
        Parameters
        ----------
        integrand: callable
            Function with input (n, d) shaped ndarrays and
            output (n,) shaped ndarray.

        Note
        ----------
        Only works for integrands which allow vectorised
        evaluations.
        """
        values = integrand(self.nodes)
        return self.weights.dot(values)


    @staticmethod
    def compute_integral(integrand, nodes, weights):
        """
        Directly computes the integral of a function.

        Parameters
        ----------
        nodes: (n, d) ndarray
            Quadrature nodes.
        weights: (n, ) ndarray
            Quadrature weights.
        integrand: callable
            Function with input (n, d) shaped ndarrays and
            output (n,) shaped ndarray.

        Note
        ----------
        Only works for integrands which allow vectorised
        evaluations.

        To-Do
        ----------
        Write a convenience function which replaces
        this staticmethod.
        """
        values = integrand(nodes)
        return weights.dot(values)

class MonteCarlo(Quadrature):
    """
    Monte Carlo quadrature rules.
    Random nodes and uniform weights.

    Attributes
    ----------
    nodes: (n, d) ndarray
        Quadrature nodes: uniform random variables.
    weights: (n, ) ndarray
        Quadrature weights, all equal to 1/N.
    """
    def __init__(self, num_pts, dim):
        """
        Initialise MC rule in given dimensionality.

        Parameters
        ----------
        num_pts: int
            Number of points.
        dim: int
            Dimensionality of the integral.
        """
        random_nodes = Random.construct(num_pts, dim)
        weights = np.ones(num_pts)/(1.0*num_pts) 
        Quadrature.__init__(self, random_nodes, weights)

    @staticmethod
    def compute_integral(integrand, num_pts, dim):
        """
        Directly computes the integral of a function
        with MC.
        Overwrites super.compute_integral()

        Parameters
        ----------
        num_pts: int
            Number of points.
        dim: int
            Dimensionality of the integral.
        integrand: callable
            Function with input (n, d) shaped ndarrays and
            output (n,) shaped ndarray.

        Note
        ----------
        Only works for integrands which allow vectorised
        evaluations.

        To-Do
        ----------
        Write a convenience function which replaces
        this staticmethod.
        """
        nodes = Random.construct(num_pts, dim)
        values = integrand(nodes)
        return np.sum(values, axis = 0)/(1.0 *num_pts)


class QuasiMonteCarlo(Quadrature):
    """
    Quasi Monte Carlo quadrature. Similar to MC
    quadrature. Quadrature nodes are not random
    but a lattice.

    Attributes
    ----------
    nodes: (n, d) ndarray
        Quadrature nodes: lattice points.
    weights: (n, ) ndarray
        Quadrature weights, all equal to 1/N.
    """
    def __init__(self, num_pts, dim):
        """
        Initialise QMC rule in given dimensionality.

        Parameters
        ----------
        num_pts: int
            Number of points.
        dim: int
            Dimensionality of the integral.
        """
        nodes = Lattice.construct(num_pts, dim)
        weights = np.ones(num_pts)/(1.0*num_pts) 
        Quadrature.__init__(self, random_nodes, weights)

    @staticmethod
    def compute_integral(integrand, num_pts, dim):
        """
        Directly computes the integral of a function
        with QMC.
        Overwrites super.compute_integral()

        Parameters
        ----------
        num_pts: int
            Number of points.
        dim: int
            Dimensionality of the integral.
        integrand: callable
            Function with input (n, d) shaped ndarrays and
            output (n,) shaped ndarray.

        Note
        ----------
        Only works for integrands which allow vectorised
        evaluations.

        To-Do
        ----------
        Write a convenience function which replaces
        this staticmethod.
        """
        nodes = Lattice.construct(num_pts, dim, rand_shift = True)
        values = integrand(nodes)
        summ = np.sum(values, axis = 0)/(1.0 *num_pts)
        return summ.reshape((1, len(summ)))





