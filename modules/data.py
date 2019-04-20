"""
NAME: data.py

PURPOSE: data class, noisy observations on locations

NOTE: We only consider additive Gaussian noise
"""

import numpy as np
from pointsets import Random, Mesh1d
from scipy import interpolate
import scipy.sparse
import scipy.sparse.linalg

class Data():

    def __init__(self, locations, true_observations, variance = 0.001):

        def make_noisy(true_observations):
            dim_obs = len(true_observations.T)
            num_obs = len(true_observations)

            noisy_observations = np.copy(true_observations)
            for i in range(num_obs):
                noise = np.sqrt(variance) * np.random.randn(1, dim_obs)
                noisy_observations[i,:] = noisy_observations[i,:] + noise
            return noisy_observations

        self.locations = locations
        self.true_observations = true_observations
        self.observations = make_noisy(true_observations)
        self.variance = variance


"""
TODO: How does it work for multidimensional data---forward map should have multidimensional output
"""
class InverseProblem(Data):

    def __init__(self, locations, forward_map, variance = 0.):
        true_observations = forward_map(locations)
        self.forward_map = forward_map
        Data.__init__(self, locations, true_observations, variance)


"""
Toy 1d ill-posed inverse problem with G(x) = sin(5x) on [0,1]
"""
class ToyInverseProblem(InverseProblem):

    """
    unpredictable behaviour for num_pts>1 because output dimension of forward map does not aligh
    """
    def __init__(self, num_pts = 1, variance = 0.01):
        
        def forward_map(points):

            def sine(pt):
                return np.sin(5*pt)

            num_pts = len(points)
            dim = len(points.T)
            assert(dim==1), "Forward map is 1D, pointset is not"

            observations = np.zeros((num_pts, 1))
            for i in range(num_pts):
                observations[i, 0] = sine(points[i, 0])
            return observations * np.ones(num_pts)

        pointset = Random(num_pts, 1)
        InverseProblem.__init__(self, pointset.points, forward_map, variance)


"""
Toy 1d Gaussian process data
"""
class ToyGPData(InverseProblem):

    def __init__(self, num_pts = 3, variance = 0.):
        
        def forward_map(points):

            def exp_sine(pt):
                return np.exp(-3*np.sin(3*pt**2))

            def forw2(pt):
                return -20*(pt-0.5)**2

            num_pts = len(points)
            dim = len(points.T)
            assert(dim==1), "Forward map is 1D, pointset is not"

            observations = np.zeros((num_pts, 1))
            for i in range(num_pts):
                observations[i, 0] = exp_sine(points[i, 0])
            return observations

        pointset = Mesh1d(num_pts)
        pointset.points = pointset.points*0.6 + 0.01
        InverseProblem.__init__(self, pointset.points, forward_map, variance)


"""
"""
class FEMInverseProblem(InverseProblem):

    """
    solves -(alpha(x,u) u'(x,a))' = 1 
    for alpha(x,u) = 0.1 * sum_{i=1}^K e^(a_i*x)
    at coefficient a with mesh width h
    and gives output measurements at pointset vec_J subset [0,1]
    """
    def __init__(self, input_dim = 1, eval_pts = np.random.rand(1,1), meshwidth = 1./32., variance = 0.001):
        
        eval_dim = len(eval_pts.T)

        def forward_map_fem(inputVec, obsPtSet = eval_pts, meshwidth = meshwidth):

            def compIntExp(coeff, lowerBd, upperBd):
                intSum = 0
                for idx in range(len(coeff)):
                    if coeff[idx] > 0:
                        intSum = intSum + (np.exp(coeff[idx]*upperBd)-np.exp(coeff[idx]*lowerBd))/coeff[idx]
                    else:
                        intSum = intSum + upperBd - lowerBd
                return 0.1 * intSum

            numNodes = int(1.0/meshwidth) + 1
            numIntNodes = numNodes - 2  

            stiffMtrxDiag = np.zeros(numIntNodes)
            stiffMtrxOffDiag = np.zeros(numIntNodes - 1)
            for idx in range(numIntNodes):
                lowBd = idx/(numIntNodes + 1.0)
                upBd = (idx + 2.0)/(numIntNodes + 1.0)
                stiffMtrxDiag[idx] = (numIntNodes + 1.0)**2 *compIntExp(inputVec,lowBd,upBd)
            for idx in range(numIntNodes-1):
                lowBd = (idx + 1.0)/(numIntNodes + 1.0)
                upBd = (idx + 2.0)/(numIntNodes + 1.0)
                stiffMtrxOffDiag[idx] = -(numIntNodes + 1.0)**2 * compIntExp(inputVec, lowBd, upBd)
            stiffMtrx = scipy.sparse.diags([stiffMtrxDiag, stiffMtrxOffDiag, stiffMtrxOffDiag], [0,-1,1], format = 'csc')

            rhs = np.ones(numIntNodes) / (numIntNodes + 1.0)

            solCoeff = scipy.sparse.linalg.spsolve(stiffMtrx,rhs)
            solCoeffWithBdry = np.zeros(numNodes)
            solCoeffWithBdry[1:(numIntNodes+1)] = solCoeff

            nodes =  np.linspace(0,1,numNodes)
            solFct = interpolate.interp1d(nodes, solCoeffWithBdry)
            return solFct(obsPtSet)

        true_input = Random(input_dim, 1)

        InverseProblem.__init__(self, true_input.points, forward_map_fem, variance)









