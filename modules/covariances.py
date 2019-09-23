"""
NAME: covariances.py

AUTHOR: NK
"""

import numpy as np
import scipy.special
import scipy.spatial

class Covariance:

    def __init__(self, cov_fct):
        self.cov_fct = cov_fct

    def evaluate(self, pointset1, pointset2, shift = 0.):
        cov_mtrx = self.cov_fct(pointset1, pointset2)
        if shift > 0:
            return cov_mtrx + shift * np.eye(len(pointset1),len(pointset2))
        else:
            return cov_mtrx




class GaussCov(Covariance):

    def __init__(self, corr_length = 1.0):

        assert(corr_length > 0), "Please enter a positive correlation length"

        def gaussian_cov(pointset1, pointset2, corr_length = corr_length):
            distance = scipy.spatial.distance_matrix(pointset1, pointset2)
            return np.exp(-distance**2/(2.0*corr_length**2))

        self.corr_length = corr_length
        Covariance.__init__(self, gaussian_cov)

    @staticmethod
    def fast_mtrx(pointset1, pointset2, corr_length = 1.0, shift = 0.):
        assert(shift >= 0), "Please enter a nonnegative shift"
        assert(corr_length >= 0), "Please enter a positive correlation length"
        cov_mtrx = scipy.spatial.distance_matrix(pointset1, pointset2)
        cov_mtrx = np.exp(-cov_mtrx**2/(2.0*corr_length**2))
        if shift > 0.0:
            return cov_mtrx + shift * np.eye(len(pointset1),len(pointset2))
        else:
            return cov_mtrx




class ExpCov(Covariance):

    def __init__(self, corr_length = 1.0):
        assert(corr_length > 0), "Please enter a positive correlation length"

        def exp_cov(pointset1, pointset2, corr_length = corr_length):
            distance = scipy.spatial.distance_matrix(pointset1, pointset2)
            return np.exp(-distance/(1.0 * corr_length))

        self.corr_length = corr_length
        Covariance.__init__(self, exp_cov)

    @staticmethod
    def fast_mtrx(pointset1, pointset2, corr_length = 1.0, shift = 0.):
        assert(corr_length > 0), "Please enter a positive correlation length"
        cov_mtrx = scipy.spatial.distance_matrix(pointset1, pointset2)
        cov_mtrx = np.exp(-cov_mtrx/(1.0*corr_length))
        if shift > 0.0:
            return cov_mtrx + shift * np.eye(len(pointset1),len(pointset2))
        else:
            return cov_mtrx



class MaternCov(Covariance):

    def __init__(self, smoothness = 1.5, corr_length = 1.0):

        assert(smoothness > 0.), "Please enter a positive smoothness parameter"
        assert(corr_length > 0.), "Please enter a positive correlation length"

        def matern_cov(pointset1, pointset2, smoothness = smoothness, corr_length = corr_length):
            distance = scipy.spatial.distance_matrix(pointset1, pointset2)
            scaled_dist_mtrx = (np.sqrt(2.0 * smoothness) * distance) / corr_length
            scaled_dist_mtrx[np.where(scaled_dist_mtrx > 0.)] = \
                2.**(1.0-smoothness) / scipy.special.gamma(smoothness) \
                * scaled_dist_mtrx[np.where(scaled_dist_mtrx > 0.)]**(smoothness) \
                * scipy.special.kv(smoothness, scaled_dist_mtrx[np.where(scaled_dist_mtrx > 0.)])
            scaled_dist_mtrx[np.where(scaled_dist_mtrx <= 0.)] = 1.0
            return scaled_dist_mtrx

        self.corr_length = corr_length
        self.smoothness = smoothness
        Covariance.__init__(self, matern_cov)

    @staticmethod
    def fast_mtrx(pointset1, pointset2, smoothness = 1.5, corr_length = 1.0, shift = 0.):

        assert(smoothness > 0.), "Please enter a positive smoothness parameter"
        assert(corr_length > 0.), "Please enter a positive correlation length"

        def matern_cov(distance, smoothness = smoothness, corr_length = corr_length):
            scaled_dist_mtrx = (np.sqrt(2.0 * smoothness) * distance) / corr_length
            scaled_dist_mtrx[np.where(scaled_dist_mtrx > 0.)] = \
                2.**(1.0-smoothness) / scipy.special.gamma(smoothness) \
                * scaled_dist_mtrx[np.where(scaled_dist_mtrx > 0.)]**(smoothness) \
                * scipy.special.kv(smoothness, scaled_dist_mtrx[np.where(scaled_dist_mtrx > 0.)])
            scaled_dist_mtrx[np.where(scaled_dist_mtrx <= 0.)] = 1.0
            return scaled_dist_mtrx

        cov_mtrx = scipy.spatial.distance_matrix(pointset1, pointset2)
        cov_mtrx = matern_cov(cov_mtrx)
        if shift > 0.0:
            return cov_mtrx + shift * np.eye(len(pointset1),len(pointset2))
        else:
            return cov_mtrx


class TpsSphereCov(Covariance):

    def __init__(self):

        def tps_cov(ptset1, ptset2):
            assert(len(ptset1.T) == len(ptset2.T))
            assert(len(ptset1.T) == 3), "Please enter a 3d pointset"
            assert(np.linalg.norm(ptset1[0,:] - 1.)), "Please use a spherical pointset"
            kmat = 0.5*scipy.spatial.distance_matrix(ptset1, ptset2)**2
            kmat = kmat * np.log(kmat + np.eye(len(ptset1), len(ptset2)))
            polblock = np.concatenate((np.ones((1, len(ptset2))), ptset2.T), axis = 0)
            kmat2 = np.concatenate((kmat, polblock), axis = 0)
            polblock1 = np.concatenate((np.ones((len(ptset1), 1)), ptset1), axis = 1)
            polblock2 = np.concatenate((polblock1, np.zeros((4,4))), axis = 0)
            return np.concatenate((kmat2, polblock2), axis = 1)

        Covariance.__init__(self, tps_cov)

    @staticmethod
    def fast_mtrx(ptset1, ptset2):
        assert(len(ptset1.T) == len(ptset2.T))
        assert(len(ptset1.T) == 3), "Please enter a 3d pointset"
        assert(np.linalg.norm(ptset1[0,:] - 1.0)), "Please use a spherical pointset"
        kmat = 0.5*scipy.spatial.distance_matrix(ptset1, ptset2)**2
        kmat = kmat * np.log(kmat + np.eye(len(ptset1), len(ptset2)))
        polblock = np.concatenate((np.ones((1, len(ptset2))), ptset2.T), axis = 0)
        kmat2 = np.concatenate((kmat, polblock), axis = 0)
        polblock1 = np.concatenate((np.ones((len(ptset1), 1)), ptset1), axis = 1)
        polblock2 = np.concatenate((polblock1, np.zeros((4,4))), axis = 0)
        return np.concatenate((kmat2, polblock2), axis = 1)


"""
Thin-plate spline function of order 2
"""
class TpsCov(Covariance):

    def __init__(self):

        def tps_cov(ptset1, ptset2):
            kmat = scipy.spatial.distance_matrix(ptset1, ptset2)
            assert(len(ptset1.T) == len(ptset2.T))  # dimensionality of pointsets aligns
            pbsize = len(ptset1.T) + 1
            kmat = scipy.spatial.distance_matrix(ptset1, ptset2)
            kmat = kmat**2 * np.log(kmat + np.finfo(float).eps)
            polblock = np.concatenate((np.ones((1, len(ptset2))), ptset2.T), axis = 0)
            kmat2 = np.concatenate((kmat, polblock), axis = 0)
            polblock1 = np.concatenate((np.ones((len(ptset1), 1)), ptset1), axis = 1)
            polblock2 = np.concatenate((polblock1, np.zeros((pbsize, pbsize))), axis = 0)
            return np.concatenate((kmat2, polblock2), axis = 1)

        Covariance.__init__(self, tps_cov)

    @staticmethod
    def fast_mtrx(ptset1, ptset2):
        kmat = scipy.spatial.distance_matrix(ptset1, ptset2)
        assert(len(ptset1.T) == len(ptset2.T))  # dimensionality of pointsets aligns
        pbsize = len(ptset1.T) + 1
        kmat = kmat**2 * np.log(kmat + np.finfo(float).eps)
        polblock = np.concatenate((np.ones((1, len(ptset2))), ptset2.T), axis = 0)
        kmat2 = np.concatenate((kmat, polblock), axis = 0)
        polblock1 = np.concatenate((np.ones((len(ptset1), 1)), ptset1), axis = 1)
        polblock2 = np.concatenate((polblock1, np.zeros((pbsize, pbsize))), axis = 0)
        return np.concatenate((kmat2, polblock2), axis = 1)


