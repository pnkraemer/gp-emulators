"""
NAME: covariances.py

PURPOSE: Covariance function for Gaussian processes

NOTE: definitions of covariances as in rasmussen/williams, chapter 4
"""
import numpy as np
from pointsets import PointSet, Random

class Covariance:

    def __init__(self, cov_fct):
        self.cov_fct = cov_fct

    def assemble_cov_mtrx(self, pointset1, pointset2):
        cov_fct = self.cov_fct
        num_pts_1 = pointset1.num_pts
        num_pts_2 = pointset2.num_pts
        points1 = pointset1.points 
        points2 = pointset2.points 

        cov_mtrx = np.zeros((num_pts_1, num_pts_2))
        for i in range(num_pts_1):
            for j in range(num_pts_2):
                cov_mtrx[i, j] = cov_fct(points1[i], points2[j])
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


