"""
NAME: means.py

PURPOSE: Mean functions for Gaussian processes

NOTE: definitions of covariances as in rasmussen/williams, chapter 4
"""
import numpy as np
from gpemu.pointsets import Random


class Mean:

    def __init__(self, mean_fct):
        self.mean_fct = mean_fct

    def evaluate(self, pointset):
        return self.mean_fct(pointset)


"""
Class for mean function
"""
class ZeroMean(Mean):

    def __init__(self):
        
        def zero_mean(ptset):
            return np.zeros((ptset.shape[0], 1))

        Mean.__init__(self, zero_mean)

class ConstMean(Mean):

    def __init__(self, const):
        
        def const_mean(ptset, const = const):
            return const * np.ones((ptset.shape[0], 1))

        Mean.__init__(self, const_mean)

