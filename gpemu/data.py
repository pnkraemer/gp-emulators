"""
NAME: data.py

AUTHOR: data class, noisy observations on locations

NOTE: We only consider additive Gaussian noise
"""
import sys
import numpy as np
from gpemu.pointsets import Random, Mesh1d
from scipy import interpolate
import scipy.sparse
import scipy.sparse.linalg

class Data():

    def __init__(self, locations, true_observations, variance = 0.):

        def make_noisy(true_observations):

            noisy_observations = np.copy(true_observations)
            if variance > 0:
                dim_obs = len(true_observations.T)
                num_obs = len(true_observations)
                for i in range(num_obs):
                    noise = np.sqrt(variance) * np.random.randn(1, dim_obs)
#                    noise = (variance) * np.random.randn(1, dim_obs)
                    noisy_observations[i,:] = noisy_observations[i,:] + noise
            return noisy_observations

        self.locations = locations
        self.true_observations = true_observations
        self.observations = make_noisy(true_observations)
        self.variance = variance

"""
An inverse problem is data together with a forward map
"""
class InverseProblem(Data):

    is_ip = True

    def __init__(self, locations, forward_map, variance = 0.):
        true_observations = forward_map(locations)
        self.forward_map = forward_map
        Data.__init__(self, locations, true_observations, variance)

"""
Toy 1d inverse problem with G(x) = sin(5x) on [0,1]
"""
class ToyInverseProblem1d(InverseProblem):

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

#           observations = np.zeros((num_pts, 1))
#           for i in range(num_pts):
#               observations[i, 0] = sine(points[i, 0])
            return sine(points).reshape((num_pts, 1)) 

        pointset = Random.construct(num_pts, 1)
        InverseProblem.__init__(self, pointset, forward_map, variance)


"""
Toy 1d Gaussian process data
"""
class ToyGPData1d(InverseProblem):

    def __init__(self, num_pts = 3, variance = 0.):
        
        def forward_map(points):

            def exp_sine(pt):
                return np.exp(-3*np.sin(3*pt**2))

            def forw2(pt):
                return -20*(pt-0.5)**2

            num_pts = len(points)
            dim = len(points.T)
            assert(dim==1), "Forward map is 1D, pointset is not"

#            observations = np.zeros((num_pts, 1))
#            for i in range(num_pts):
#                observations[i, 0] = exp_sine(points[i, 0])
            return exp_sine(points).reshape((num_pts, 1))

        pointset = Mesh1d.construct(num_pts)
        pointset = pointset*0.6 + 0.01
        InverseProblem.__init__(self, pointset, forward_map, variance)


"""
FEM IP
"""
class FEMInverseProblem(InverseProblem):

    """
    solves -(alpha(x,a) u'(x,a))' = 1 
    for alpha(x,a) = 1 + 0.1 * sum_{i=1}^K sin(a_i*x)
    """
    def __init__(self, input_dim = 1, eval_pts = np.random.rand(1,1), meshwidth = 1./32., variance = 0.001):
        
        eval_dim = len(eval_pts.T)

        def forward_map_fem(inputVec, obsPtSet = eval_pts, meshwidth = meshwidth):

            def compIntExp(coeff, lowerBd, upperBd):
                intSum = 0
                coeff = coeff[0]
                for idx in range(len(coeff)):
                    if coeff[idx] > 0:
                        intSum = intSum + (np.cos(coeff[idx]*lowerBd)-np.cos(coeff[idx]*upperBd))/coeff[idx]
                return (upperBd - lowerBd) * 1.0 + 0.1 * intSum

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
            return solFct(obsPtSet).reshape((1, len(obsPtSet)))
        
        def forward_map(locations):
            evaluations = np.zeros((len(locations), 1))
            for i in range(len(locations)):
                evaluations[i,0] = forward_map_fem(locations[i,:].reshape((1,len(locations.T))))
            return evaluations




        true_input = Random.construct(1, input_dim)

        InverseProblem.__init__(self, true_input, forward_map, variance)









