
from gpemu.pointsets import *

from scipy.stats import norm
import numpy as np


class Prior():

    @staticmethod
    def uniform(locations):
        return np.ones((len(locations), 1))

    @staticmethod
    def gaussian1d(locations, mean = 0.5, variance = 0.15):

        return norm.pdf(locations[:,0], mean, variance).reshape((len(locations),1))

    @staticmethod
    def gaussian1d_mix(locations, mean1 = 0.3, variance1 = 0.05, mean2 = 0.7, variance2 = 0.1):

        return 0.5 * norm.pdf(locations[:,0], mean1, variance1).reshape((len(locations),1)) + 0.5 * norm.pdf(locations[:,0], mean2, variance2).reshape((len(locations),1))
