"""
NAME: covariances.py

PURPOSE: Covariance function for Gaussian processes

NOTE: definitions of covariances as in rasmussen/williams, chapter 4
"""
from __future__ import division
import numpy as np
import scipy.special

from pointsets import PointSet


class Covariance:

    def __init__(self, cov_fct):
        self.cov_fct = cov_fct

    def assemble_cov_mtrx(self, pointset1, pointset2, shift = 0.):
        cov_fct = self.cov_fct
        num_pts_1 = pointset1.num_pts
        num_pts_2 = pointset2.num_pts
        points1 = pointset1.points 
        points2 = pointset2.points 

        cov_mtrx = np.zeros((num_pts_1, num_pts_2))
        for i in range(num_pts_1):
            for j in range(num_pts_2):
                cov_mtrx[i, j] = cov_fct(points1[i], points2[j])
                if i==j:
                    cov_mtrx[i,j] = cov_mtrx[i,j] + shift
        return cov_mtrx


class GaussCov(Covariance):

    def __init__(self, corr_length = 1.0):

        def gaussian_cov(pt1, pt2, corr_length = corr_length):
            norm_of_diff = np.linalg.norm(pt1 - pt2)
            return np.exp(-norm_of_diff**2/(2*corr_length**2))

        self.corr_length = corr_length
        Covariance.__init__(self, gaussian_cov)


class ExpCov(Covariance):

    def __init__(self, corr_length = 1.0):

        def exp_cov(pt1, pt2, corr_length = corr_length):
            norm_of_diff = np.linalg.norm(pt1 - pt2)
            return np.exp(-norm_of_diff/(corr_length))

        self.corr_length = corr_length
        Covariance.__init__(self, exp_cov)


class MaternCov(Covariance):

    def __init__(self, smoothness = 1.5, corr_length = 1.0):

        def matern_cov(pt1, pt2, smoothness = smoothness, corr_length = corr_length):
            norm_of_diff = np.linalg.norm(pt1 - pt2)
            if norm_of_diff <= 0:
                return 1.0
            else:
                scaled_norm = np.sqrt(2.0 * smoothness) * norm_of_diff / corr_length
                return  2**(1.0-smoothness) / scipy.special.gamma(smoothness) * norm_of_diff**(smoothness) * scipy.special.kv(smoothness, norm_of_diff)

        self.corr_length = corr_length
        self.smoothness = smoothness
        Covariance.__init__(self, matern_cov)


# from pointsets import Random

# ptset = Random(100, 1)
# ptset.construct_pointset()



# matern_cov = MaternCov(2.0, 1.0)
# A = matern_cov.assemble_cov_mtrx(ptset, ptset)


# print(A)

