"""
NAME: data.py

PURPOSE: data class: noisy observations on locations

NOTE: We only consider additive Gaussian noise
"""

import numpy as np
from pointsets import Random, Mesh1d

class Data():

    # Assume additive Gaussian observation error -> std_dev
    def __init__(self, locations, true_observations, variance = 0.):

        def make_noisy(true_observations):
            dim_observations = len(true_observations.T)
            num_pts = len(true_observations)

            noisy_observations = np.copy(true_observations)
            for i in range(num_pts):
                noise = np.sqrt(variance) * np.random.randn(1, dim_observations)
                noisy_observations[i,:] = noisy_observations[i,:] + noise
            return noisy_observations

        self.locations = locations
        self.true_observations = true_observations
        self.observations = make_noisy(true_observations)
        self.variance = variance



class InverseProblem(Data):

    def __init__(self, locations, forward_map, variance = 0.):
        true_observations = forward_map(locations)
        self.forward_map = forward_map
        Data.__init__(self, locations, true_observations, variance)


"""
Toy 1d ill-posed inverse problem with G(x) = sin(5x) on [0,1]

TODO: evaluate forward map function
"""
class ToyInverseProblem(InverseProblem):


    def __init__(self, variance = 0.):
        
        def forward_map(locations):

            def sine(pt):
                return np.sin(5*pt)

            points = locations.points
            num_pts = locations.num_pts
            dim = locations.dim
            assert(dim==1), "Forward map is 1D, pointset is not"

            observations = np.zeros((num_pts, 1))
            for i in range(num_pts):
                observations[i, 0] = sine(points[i, 0])
            return observations

        pointset = Random(1, 1)
        InverseProblem.__init__(self, pointset, forward_map, variance)


class ToyGPData(InverseProblem):

    def __init__(self, num_pts = 3, variance = 0.):
        
        def forward_map(locations):

            def exp_sine(pt):
                return np.exp(-np.sin(10*pt)**2)

            points = locations.points
            num_pts = locations.num_pts
            dim = locations.dim
            assert(dim==1), "Forward map is 1D, pointset is not"

            observations = np.zeros((num_pts, 1))
            for i in range(num_pts):
                observations[i, 0] = exp_sine(points[i, 0])
            return observations

        pointset = Mesh1d(num_pts)
        InverseProblem.__init__(self, pointset, forward_map, variance)





# IP = ToyInverseProblem(0.0001)
# print(IP.locations.points)
# print(IP.true_observations)
# print(IP.noisy_observations)
# print(IP.variance)


# print()
# B = Random(20,1)
# print(IP.forward_map(B))





