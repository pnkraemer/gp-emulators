"""
NAME: means.py

PURPOSE: Mean functions for Gaussian processes

NOTE: definitions of covariances as in rasmussen/williams, chapter 4
"""
import numpy as np
from pointsets import PointSet, Random


class Mean:

    def __init__(self, mean_fct):
        self.mean_fct = mean_fct

    def assemble_mean_vec(self, points):
  #      print("pt shape =", points.shape)
        mean_fct = self.mean_fct
        mean_vec = mean_fct(points)
        return mean_vec

    def evaluate(self, points):
        return self.mean_fct(points)


"""
Class for mean function
"""
class ZeroMean(Mean):
    def __init__(self, corr_length = 1.0):
        
        def zero_mean(pt):
            return np.zeros((pt.shape[0], 1))

        Mean.__init__(self, zero_mean)

