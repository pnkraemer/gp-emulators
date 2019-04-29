"""
NAME: posterior.py

NOTE: We only consider additive Gaussian noise
"""

import numpy as np
import sys

sys.path.insert(0, "../../modules")
from pointsets import *
from data import Data, FEMInverseProblem
from quadrature import MonteCarlo
from means import ZeroMean
from covariances import MaternCov
from gaussianprocesses import GaussianProcess, ConditionedGaussianProcess

class Posterior():
	def __init__(self, inverse_problem, prior_density):
		self.ip = inverse_problem
		self.norm_const = None
		self.prior_density = prior_density

	def potential(self, locations):
		diff = np.zeros(len(locations))
		for i in range(len(locations)):
			diff[i] = np.linalg.norm(self.ip.observations - self.ip.forward_map(locations[i,:].reshape((1, len(locations.T)))))**2
		return diff

	def likelihood(self, locations):
		return np.exp(-self.potential(locations)/(2*self.ip.variance))

	def compute_norm_const(self, num_mc_pts = 10000):

		def integrand(locations):
			return self.likelihood(locations) * self.prior_density(locations)

		num_true_inputs = len(self.ip.locations)
		self.norm_const = MonteCarlo.compute_integral(integrand, num_mc_pts, num_true_inputs)

	def density(self, locations):
		if self.norm_const is None:
			print("Computing normalisation constant on N = 10000 pts...", end = "")
			self.compute_norm_const(10000)	
			print("done!")	
		return self.likelihood(locations)/self.norm_const * self.prior_density(locations)




class ApproximatePosterior(Posterior):

	def __init__(self, posterior, gp):
		Posterior.__init__(self, posterior.ip)
		self.gp = gp
		self.posterior = posterior

	def potential2(self, locations):
		diff = np.zeros(len(locations))
		for i in range(len(locations)):
			evaluate = self.gp.mean_fct.evaluate(np.array([locations[i,:]]))
			diff[i] = np.linalg.norm(self.ip.observations - evaluate)**2
		return diff

	"""
	num_observations is output dimension of forward model
	"""
	def makedata(self, locations, approximand, num_observations = 1):
		ip = self.posterior.ip
		observations = np.zeros((len(locations), num_observations))
		for i in range(len(locations)):
			observations[i,:] = approximand(locations[i,:].reshape((1, len(locations.T))))
		self.approx_data = Data(locations, observations, 0.0)
		self.gp = ConditionedGaussianProcess(self.gp, self.approx_data)

	def approximate_forwardmap(self, pointset, num_observations = 1):
		assert(self.gp.is_conditioned == False), "Approximation already in use! Make new posterior"
		self.makedata(pointset, self.ip.forward_map)
		self.potential = self.potential2

	def potential3(self, locations):
		diff = np.zeros(len(locations))
		for i in range(len(locations)):
			evaluate = self.gp.mean_fct.evaluate(np.array([locations[i,:]]))
			diff[i] = evaluate
		return diff

	def approximate_potential(self, pointset):
		assert(self.gp.is_conditioned == False), "Approximation already in use! Make new posterior"
		self.makedata(pointset, self.posterior.potential)
		self.potential = self.potential3


	def likelihood2(self, locations):
		approx_potent = np.zeros(len(locations))
		for i in range(len(locations)):
			evaluate = self.gp.mean_fct.evaluate(np.array([locations[i,:]]))
			approx_potent[i] = evaluate
		return approx_potent

	def approximate_likelihood(self, pointset):
		assert(self.gp.is_conditioned == False), "Approximation already in use! Make new posterior"
		self.makedata(pointset, self.posterior.likelihood)
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












