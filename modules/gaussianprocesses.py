"""
NAME: gaussianprocesses.py

PURPOSE: Gaussian process class

TODO: new_mean and new_cov should rather be instances mean and cov,
the conditioned gp should be an instance of gp 
(this is 1) cleaner and 2) eases repeated applications of regression)
"""
import numpy as np 

from covariances import *
from means import *
from data import Data
import sys
import scipy.linalg

"""
A Gaussian process is a mean function and a covariance function
"""
class GaussianProcess:

    def __init__(self, mean_fct, cov_fct):
        self.mean_fct = mean_fct
        self.cov_fct = cov_fct
        self.is_conditioned = False

    def sample(self, sample_locations):
        mean_vec = self.mean_fct.evaluate(sample_locations)
        cov_mtrx = self.cov_fct.evaluate(sample_locations, sample_locations)
        return np.random.multivariate_normal(mean_vec.reshape([len(mean_vec),]), cov_mtrx, 1).T

    def sample_many(self, sample_locations, num_samps):
        mean_vec = self.mean_fct.evaluate(sample_locations)
        cov_mtrx = self.cov_fct.evaluate(sample_locations, sample_locations)
        return np.random.multivariate_normal(mean_vec.reshape([len(mean_vec),]), cov_mtrx, num_samps).T


class StandardGP(GaussianProcess):

    def __init__(self):
        zero_mean = ZeroMean()
        matern_cov = MaternCov()
        GaussianProcess.__init__(self, zero_mean, matern_cov)





"""
Conditioned Gaussian process class
->Result of GP regression

A conditioned GP is a GP with added data set $(X,y)$, 
an inverse covariance matrix $k(X,X)^{-1}$ and corresponding 
mean and covariance functions based on the these

"""
class ConditionedGaussianProcess(GaussianProcess):

    def __init__(self, GaussProc, data):
        self.data = data
        cov_mtrx = GaussProc.cov_fct.evaluate(self.data.locations, self.data.locations, self.data.variance)
        self.cov_mtrx = cov_mtrx
        self.prior = GaussProc
        
        def new_mean_fct(loc, data = self.data, GP = GaussProc, cm = self.cov_mtrx):
            mean_vec_oldloc = GP.mean_fct.evaluate(data.locations)
            mean_vec_newloc = GP.mean_fct.evaluate(loc)
            cov_mtrx = GP.cov_fct.evaluate(loc, data.locations)
            obs2 = data.observations - mean_vec_oldloc
            lu, piv = scipy.linalg.lu_factor(cm)
            coeff = scipy.linalg.lu_solve((lu, piv), obs2)
#            coeff = np.linalg.solve(cm, obs2)
            return mean_vec_newloc + cov_mtrx.dot(coeff)

        def new_cov_fct(loc1, loc2, data = self.data, GP = GaussProc, cm = self.cov_mtrx):
            cov_mtrx_new = GP.cov_fct.evaluate(loc1, loc2)
            cov_mtrx_new2 = GP.cov_fct.evaluate(loc1, data.locations)
            cov_mtrx_new3 = GP.cov_fct.evaluate(data.locations, loc2)
            lu, piv = scipy.linalg.lu_factor(cm)
            coeff = scipy.linalg.lu_solve((lu, piv), cov_mtrx_new3)
#            coeff = np.linalg.solve(cm, cov_mtrx_new3)
            return cov_mtrx_new - cov_mtrx_new2.dot(coeff)

        GaussianProcess.__init__(self, Mean(new_mean_fct), Covariance(new_cov_fct))
        self.is_conditioned = True


