"""
NAME: gaussianprocesses.py

PURPOSE: Gaussian process class

"""
import numpy as np 

from covariances import Covariance
from means import Mean
from data import Data
from pointsets import PointSet
import sys


"""
Gaussian process class

ATTRIBUTES: 
    mean function: mean_fct from mean class
    covariance function: cov_fct from covariance class
"""
class GaussianProcess:
    def __init__(self, mean_fct, cov_fct):
        self.mean_fct = mean_fct
        self.cov_fct = cov_fct

    def sample(self, sample_locations):
        mean_fct = self.mean_fct
        cov_fct = self.cov_fct

        mean_vec = mean_fct.assemble_mean_vec(sample_locations.points)
      #  print("mv shape=", mean_vec.shape)
        cov_mtrx = cov_fct.assemble_cov_mtrx(sample_locations.points, sample_locations.points)
        # print("mean =", mean_vec)

        return np.random.multivariate_normal(mean_vec.reshape([len(mean_vec),]), cov_mtrx)






"""
Conditioned Gaussian process class
->Result of GP regression

ADDITIONAL ATTRIBUTES: 
    training points: data from data class
    (K + sigma^2 I)^{-1}: inv_cov_mtrx, a matrix
"""
class ConditionedGaussianProcess(GaussianProcess):

    def __init__(self, GaussProc, data):
        self.data = data
        cov_mtrx = GaussProc.cov_fct.assemble_cov_mtrx(self.data.locations.points, self.data.locations.points, self.data.variance)
        inv_cov_mtrx = np.linalg.inv(cov_mtrx)

        def new_mean_fct(loc, data = self.data, GP = GaussProc, inv_cov = inv_cov_mtrx):
            mean_vec_oldloc = GP.mean_fct.assemble_mean_vec(data.locations.points)
            mean_vec_newloc = GP.mean_fct.assemble_mean_vec(loc)
            cov_mtrx = GP.cov_fct.assemble_cov_mtrx(loc, data.locations.points)
            obs2 = data.observations - mean_vec_oldloc
            return mean_vec_newloc + cov_mtrx.dot(inv_cov.dot(obs2))

        def new_cov_fct(loc1, loc2, data = self.data, GP = GaussProc, inv_cov = inv_cov_mtrx):
            cov_mtrx_new = GP.cov_fct.assemble_cov_mtrx(loc1, loc2)
            cov_mtrx_new2 = GP.cov_fct.assemble_cov_mtrx(loc1, data.locations.points)
            cov_mtrx_new3 = GP.cov_fct.assemble_cov_mtrx(data.locations.points, loc2)
            return cov_mtrx_new - cov_mtrx_new2.dot(inv_cov).dot(cov_mtrx_new3)

        GaussianProcess.__init__(self, Mean(new_mean_fct), Covariance(new_cov_fct))


