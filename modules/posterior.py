"""
NAME: likelihood.py

PURPOSE: likelihood functions, inverse problem info plus methods

NOTE: We only consider additive Gaussian noise
"""

import numpy as np
import sys

sys.path.insert(0, "../../modules")
from pointsets import PointSet, Random, Mesh1d
from data import FEMInverseProblem
from quadrature import MonteCarlo


class Posterior():
	def __init__(self, inverse_problem):
		self.ip = inverse_problem
		self.norm_const = None

	def potential(self, locations):
		diff = np.zeros(len(locations))
		for i in range(len(locations)):
			diff[i] = np.linalg.norm(self.ip.observations - self.ip.forward_map(locations[i,:]))**2
		return diff

	def likelihood(self, locations):
		return np.exp(-self.potential(locations)/(2*self.ip.variance))

	def compute_norm_const(self, num_mc_pts = 10000):
		num_true_inputs = len(self.ip.locations)
		self.norm_const = MonteCarlo.fast_approximate(num_mc_pts, num_true_inputs, self.likelihood)

	def density(self, locations):
		if self.norm_const is None:
			print("Computing normalisation constant on N = 10000 pts...", end = "")
			self.compute_norm_const(10000)	
			print("done!")	
		return self.likelihood(locations)/self.norm_const









# num_true_inputs = 1
# eval_pts = np.array([[0.5]])
# meshwidth = 1./32.
# variance = 1e-8
# fem_ip = FEMInverseProblem(num_true_inputs, eval_pts, meshwidth, variance)
# print(fem_ip.locations)
# #print(fem_ip.observations)
# #print(fem_ip.true_observations)
# #print(fem_ip.variance)

# posterior = Posterior(fem_ip)
# posterior.compute_norm_const()
# print(posterior.norm_const)

# A = Mesh1d(500)
# B = posterior.posterior_density(A.points)
# import matplotlib.pyplot as plt 
# plt.style.use("ggplot")
# plt.plot(A.points, B)
# plt.show()












