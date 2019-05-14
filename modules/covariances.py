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
            if np.array_equal(pointset1, pointset2) == 1:#, "Shifting inappropriate for different pointsets"
                return cov_mtrx + shift * np.identity(len(pointset1))
        else:
            return cov_mtrx




class GaussCov(Covariance):

    def __init__(self, corr_length = 1.0):

        def gaussian_cov(pointset1, pointset2, corr_length = corr_length):
            distance = scipy.spatial.distance_matrix(pointset1, pointset2)
            return np.exp(-distance**2/(2.0*corr_length**2))

        self.corr_length = corr_length
        Covariance.__init__(self, gaussian_cov)

    @staticmethod
    def fast_mtrx(pointset1, pointset2, corr_length = 1.0):
        cov_mtrx = scipy.spatial.distance_matrix(pointset1, pointset2)
        return np.exp(-cov_mtrx**2/(2.0*corr_length**2))




class ExpCov(Covariance):

    def __init__(self, corr_length = 1.0):

        def exp_cov(pointset1, pointset2, corr_length = corr_length):
            distance = scipy.spatial.distance_matrix(pointset1, pointset2)
            return np.exp(-distance/(1.0 * corr_length))

        self.corr_length = corr_length
        Covariance.__init__(self, exp_cov)

    @staticmethod
    def fast_mtrx(pointset1, pointset2, corr_length = 1.0):
        cov_mtrx = scipy.spatial.distance_matrix(pointset1, pointset2)
        return np.exp(-cov_mtrx/(1.0*corr_length))



class MaternCov(Covariance):

    def __init__(self, smoothness = 1.5, corr_length = 1.0):

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
    def fast_mtrx(pointset1, pointset2, smoothness = 1.5, corr_length = 1.0):

            def matern_cov(distance, smoothness = smoothness, corr_length = corr_length):
                scaled_dist_mtrx = (np.sqrt(2.0 * smoothness) * distance) / corr_length
                scaled_dist_mtrx[np.where(scaled_dist_mtrx > 0.)] = \
                    2.**(1.0-smoothness) / scipy.special.gamma(smoothness) \
                    * scaled_dist_mtrx[np.where(scaled_dist_mtrx > 0.)]**(smoothness) \
                    * scipy.special.kv(smoothness, scaled_dist_mtrx[np.where(scaled_dist_mtrx > 0.)])
                scaled_dist_mtrx[np.where(scaled_dist_mtrx <= 0.)] = 1.0
                return scaled_dist_mtrx

            cov_mtrx = scipy.spatial.distance_matrix(pointset1, pointset2)
            return matern_cov(cov_mtrx)


class TPSSphere(Covariance):

    def __init__(self):

        def tps_cov(pointset1, pointset2):
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
        kmat = 0.5*scipy.spatial.distance_matrix(ptset1, ptset2)**2
        kmat = kmat * np.log(kmat + np.eye(len(ptset1), len(ptset2)))
        polblock = np.concatenate((np.ones((1, len(ptset2))), ptset2.T), axis = 0)
        kmat2 = np.concatenate((kmat, polblock), axis = 0)
        polblock1 = np.concatenate((np.ones((len(ptset1), 1)), ptset1), axis = 1)
        polblock2 = np.concatenate((polblock1, np.zeros((4,4))), axis = 0)
        return np.concatenate((kmat2, polblock2), axis = 1)



class TPS(Covariance):

    def __init__(self):

        def tps_cov(pointset1, pointset2):
            kmat = scipy.spatial.distance_matrix(ptset1, ptset2)
            kmat = kmat * np.log(kmat + np.eye(len(ptset1), len(ptset2)))
            polblock = np.concatenate((np.ones((1, len(ptset2))), ptset2.T), axis = 0)
            kmat2 = np.concatenate((kmat, polblock), axis = 0)
            polblock1 = np.concatenate((np.ones((len(ptset1), 1)), ptset1), axis = 1)
            polblock2 = np.concatenate((polblock1, np.zeros((3,3))), axis = 0)
            return np.concatenate((kmat2, polblock2), axis = 1)

        Covariance.__init__(self, tps_cov)

    @staticmethod
    def fast_mtrx(ptset1, ptset2):
        kmat = scipy.spatial.distance_matrix(ptset1, ptset2)
        kmat = kmat * np.log(kmat + np.eye(len(ptset1), len(ptset2)))
        polblock = np.concatenate((np.ones((1, len(ptset2))), ptset2.T), axis = 0)
        kmat2 = np.concatenate((kmat, polblock), axis = 0)
        polblock1 = np.concatenate((np.ones((len(ptset1), 1)), ptset1), axis = 1)
        polblock2 = np.concatenate((polblock1, np.zeros((3,3))), axis = 0)
        return np.concatenate((kmat2, polblock2), axis = 1)


