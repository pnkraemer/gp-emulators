"""
NAME: gaussianprocesses.py

PURPOSE: Gaussian process class

unfinished! Make "discrete GP" class?
"""
import numpy as np 

from covariances import Covariance
from means import Mean
from data import Data

class GaussianProcess:

    def __init__(self, mean_fct, cov_fct):
        self.mean_fct = mean_fct
        self.cov_fct = cov_fct

    def draw(self, locations):
        mean_fct = self.mean_fct
        cov_fct = self.cov_fct

        mean_vec = mean_fct.assemble_mean_vec(locations)
        cov_mtrx = cov_fct.assemble_cov_mtrx(locations, locations)
        return np.random.multivariate_normal(mean_vec, cov_mtrx)

    # unfinished; where does the coefficient knowledge belong?
    def condition(self, data):
    	self.data = data
    	locations = self.data.locations
    	observations = self.data.observations
    	variance = self.data.variance


    	mean_vec = mean_fct.assemble_mean_vec(locations)
        cov_mtrx = cov_fct.assemble_cov_mtrx(locations, locations, variance)

        coeff = np.linalg.solve(cov_mtrx, observations - mean_vec)


from pointsets import PointSet, Random
from covariances import GaussCov, ExpCov, MaternCov
from means import ZeroMean

import matplotlib.pyplot as plt 

zero_mean = ZeroMean()
gauss_cov = MaternCov()

gp_gauss = GaussianProcess(zero_mean, gauss_cov)
num_pts = 200
dim = 1
num_plots = 5

random_pts = Random(num_pts, dim)
random_pts.construct_pointset()
plt.style.use("ggplot")
for i in range(num_plots):
    samples = gp_gauss.draw(random_pts)
    plt.plot(random_pts.points, samples, 'o')
plt.show()

















