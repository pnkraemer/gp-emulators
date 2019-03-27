"""
NAME: covariances.py

PURPOSE: Covariance function for Gaussian processes

Definitions of covariances as in rasmussen/williams, chapter 4
"""

from __future__ import division
import numpy as np
import scipy.special
import scipy.spatial

from pointsets import PointSet


class Covariance:

    def __init__(self, cov_fct):
        self.cov_fct = cov_fct

    def assemble_cov_mtrx(self, points1, points2, shift = 0.):
        cov_mtrx = self.cov_fct(points1, points2)
        if shift > 0:
            assert(np.array_equal(points1, points2) == 1), "Shift inappropriate for different pointsets"
            cov_mtrx = cov_mtrx + shift * np.identity(len(points1))
        return cov_mtrx

    def evaluate(self, pt1, pt2):
        return self.cov_fct(pt1, pt2)


class GaussCov(Covariance):

    def __init__(self, corr_length = 1.0):

        def gaussian_cov(pt1, pt2, corr_length = corr_length):
            dist_mtrx = scipy.spatial.distance_matrix(pt1, pt2)
            return np.exp(-dist_mtrx**2/(2*corr_length**2))

        self.corr_length = corr_length
        Covariance.__init__(self, gaussian_cov)


class ExpCov(Covariance):

    def __init__(self, corr_length = 1.0):

        def exp_cov(pt1, pt2, corr_length = corr_length):
            dist_mtrx = scipy.spatial.distance_matrix(pt1, pt2)
            return np.exp(-dist_mtrx/(1.0 * corr_length))

        self.corr_length = corr_length
        Covariance.__init__(self, exp_cov)


class MaternCov(Covariance):

    def __init__(self, smoothness = 1.5, corr_length = 1.0):

        def matern_cov(pt1, pt2, smoothness = smoothness, corr_length = corr_length):
            dist_mtrx = scipy.spatial.distance_matrix(pt1, pt2)
            scaled_dist_mtrx = (np.sqrt(2.0 * smoothness) * dist_mtrx) / corr_length
            scaled_dist_mtrx[np.where(scaled_dist_mtrx > 0.)] = \
                2.**(1.0-smoothness) / scipy.special.gamma(smoothness) \
                * scaled_dist_mtrx[np.where(scaled_dist_mtrx > 0.)]**(smoothness) \
                * scipy.special.kv(smoothness, scaled_dist_mtrx[np.where(scaled_dist_mtrx > 0.)])
            scaled_dist_mtrx[np.where(scaled_dist_mtrx <= 0.)] = 1.0
            return scaled_dist_mtrx

        self.corr_length = corr_length
        self.smoothness = smoothness
        Covariance.__init__(self, matern_cov)


