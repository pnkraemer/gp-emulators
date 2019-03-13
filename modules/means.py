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

    def assemble_mean_vec(self, pointset):
        mean_fct = self.mean_fct
        num_pts = pointset.num_pts
        dim = pointset.dim
        points = pointset.points 
 #       print("A =", points.shape)
  #      print("A2 =", num_pts)
        mean_vec = np.zeros(num_pts)
        for i in range(num_pts):
   #         print("B =", points[i].shape)
    #        print("B2 =", points[i].reshape([1, dim]).shape, i)
  #          print(points[i].reshape([1, dim]))
            mean_vec[i] = mean_fct(points[i].reshape([1, dim]))
        return mean_vec


class ZeroMean(Mean):

    def __init__(self, corr_length = 1.0):

        def zero_mean(pt):
            return 0

        Mean.__init__(self, zero_mean)

