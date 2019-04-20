"""
NAME: likelihood.py

PURPOSE: likelihood functions, inverse problem info plus methods

NOTE: We only consider additive Gaussian noise
"""

import numpy as np
import sys

sys.path.insert(0, "../../modules")
from pointsets import PointSet, Random, Mesh1d
from data import Data, FEMInverseProblem
from quadrature import MonteCarlo
from means import ZeroMean
from covariances import MaternCov
from gaussianprocesses import GaussianProcess, ConditionedGaussianProcess

class Posterior():
	def __init__(self, inverse_problem):
		self.ip = inverse_problem
		self.norm_const = None

	def potential(self, locations):
		locations = locations.reshape((len(locations),1))
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




class ApproximatePosterior(Posterior):

	def __init__(self, posterior, prior_gp):
		Posterior.__init__(self, posterior.ip)
		self.prior_gp = prior_gp
		self.posterior = posterior

	def potential2(self, locations):
		diff = np.zeros(len(locations))
		for i in range(len(locations)):
			evaluate = self.cond_gp.mean_fct.evaluate(np.array([locations[i,:]]))
			diff[i] = np.linalg.norm(self.ip.observations - evaluate)**2
		return diff

	def makedata(self, locations, approximand, num_observations = 1):
		ip = self.posterior.ip
		observations = np.zeros((len(locations), num_observations))
		for i in range(len(locations)):
			observations[i,:] = approximand(locations[i,:])
		self.approx_data = Data(locations, observations, 0.0)
		self.cond_gp = ConditionedGaussianProcess(self.prior_gp, self.approx_data)


	def approximate_forwardmap(self, pointset, num_observations):
		self.makedata(pointset.points, ip.forward_map)
		self.potential = self.potential2

	def potential3(self, locations):
		diff = np.zeros(len(locations))
		for i in range(len(locations)):
			evaluate = self.cond_gp.mean_fct.evaluate(np.array([locations[i,:]]))
			diff[i] = evaluate
		return diff

	def approximate_potential(self, pointset):
		self.makedata(pointset.points, self.posterior.potential)
		self.potential = self.potential3


	def likelihood2(self, locations):
		approx_potent = np.zeros(len(locations))
		for i in range(len(locations)):
			evaluate = self.cond_gp.mean_fct.evaluate(np.array([locations[i,:]]))
			approx_potent[i] = evaluate
		return approx_potent

	def approximate_likelihood(self, pointset):
		self.makedata(pointset.points, self.posterior.likelihood)
		self.likelihood = self.likelihood2








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












