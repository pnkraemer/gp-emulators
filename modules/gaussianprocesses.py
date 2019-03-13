"""
NAME: gaussianprocesses.py

PURPOSE: Gaussian process class

unfinished! Make "discrete GP" class?
"""
import numpy as np 

from covariances import Covariance
from means import Mean
from data import Data
from pointsets import PointSet

class GaussianProcess:
    """
    Gaussian process class

    ATTRIBUTES: 
        mean function: mean_fct
        covariance function: cov_fct
    """
    def __init__(self, mean_fct, cov_fct):
        self.mean_fct = mean_fct
        self.cov_fct = cov_fct

    def sample(self, sample_locations):
        mean_fct = self.mean_fct
        cov_fct = self.cov_fct

        mean_vec = mean_fct.assemble_mean_vec(sample_locations)
        cov_mtrx = cov_fct.assemble_cov_mtrx(sample_locations, sample_locations)
        return np.random.multivariate_normal(mean_vec, cov_mtrx)

"""
TO BE FINISHED:
"""
class ConditionedGaussianProcess(GaussianProcess):
    """
    Conditioned Gaussian process class
    Result of GP regression

    ADDITIONAL ATTRIBUTES: 
        training points: data
        (K + sigma^2 I)^{-1}: inv_cov_mtrx
    """

    def __init__(self, GaussProc, data):
        GaussianProcess.__init__(self, GaussProc.mean_fct, GaussProc.cov_fct)
        self.data = data
        mean_vec = GaussProc.mean_fct.assemble_mean_vec(data.locations)
        cov_mtrx = GaussProc.cov_fct.assemble_cov_mtrx(data.locations, data.locations, data.variance)
        inv_cov_mtrx = np.linalg.inv(cov_mtrx)

        def new_mean_fct(pt, data = data, GP = GaussProc, inv_cov = inv_cov_mtrx):
            loc = PointSet.make_pointset(pt)
            mean_vec_oldloc = GP.mean_fct.assemble_mean_vec(data.locations)
            mean_vec_newloc = GP.mean_fct.assemble_mean_vec(loc)
            cov_mtrx = GP.cov_fct.assemble_cov_mtrx(loc, data.locations)
            obs2 = data.observations.T - mean_vec_oldloc
  #          print(inv_cov.dot(obs2.T))
            return mean_vec_newloc + cov_mtrx.dot(inv_cov.dot(obs2.T))
        self.mean_fct = Mean(new_mean_fct)

        def new_cov_fct(pt1, pt2, data = data, GP = GaussProc, inv_cov = inv_cov_mtrx):
            loc1 = PointSet.make_pointset(pt1)
            loc2 = PointSet.make_pointset(pt2)
            cov_mtrx_new = GP.cov_fct.assemble_cov_mtrx(loc1, loc2)
            cov_mtrx_new2 = GP.cov_fct.assemble_cov_mtrx(loc1, data.locations)
            cov_mtrx_new3 = GP.cov_fct.assemble_cov_mtrx(data.locations, loc2)
            return cov_mtrx_new - cov_mtrx_new2.dot(inv_cov).dot(cov_mtrx_new3)
        self.cov_fct = Covariance(new_cov_fct)





    def condition(self, data):
        # assert(data.locations == self.locations), "data locations do not fit"
        if data.variance > 0:
            cov_mtrx = self.cov_mtrx + np.identity(self.locations.num_pts)
        else:
            cov_mtrx = self.cov_mtrx
        self.coeff = np.linalg.solve(cov_mtrx, data.observations)

"""
(END OF TO BE FINISHED)
"""















    # # unfinished; where does the coefficient knowledge belong?
    # def condition(self, data):
    #     self.data = data
    # 	locations = self.data.locations
    # 	observations = self.data.observations
    # 	variance = self.data.variance


    # 	mean_vec = mean_fct.assemble_mean_vec(locations)
    #     cov_mtrx = cov_fct.assemble_cov_mtrx(locations, locations, variance)

    #     coeff = np.linalg.solve(cov_mtrx, observations - mean_vec)



'''
Some testing
'''
from pointsets import PointSet, Random, Mesh1d
from covariances import GaussCov, ExpCov, MaternCov
from means import ZeroMean
from data import ToyGPData

import matplotlib.pyplot as plt 

zero_mean = ZeroMean()
cov_fct = MaternCov(1.5)

gp = GaussianProcess(zero_mean, cov_fct)

num_pts = 100
dim = 1
data = ToyGPData(4)
cGP = ConditionedGaussianProcess(gp, data)




# print("data locations:", data.locations.points)

num_plots = 10
mesh = Mesh1d(num_pts)
# print("mesh: ", mesh.points)
plt.style.use("ggplot")
for i in range(num_plots):
    samples = cGP.sample(mesh)
    plt.plot(mesh.points, samples, linewidth = 2)

plt.plot(data.locations.points, data.observations, 'o', color = "black")
plt.show()

















