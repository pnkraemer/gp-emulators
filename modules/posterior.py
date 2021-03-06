"""
NAME: posterior.py

NOTE: We only consider additive Gaussian noise
"""

import numpy as np
import sys

sys.path.insert(0, "../../modules")
from pointsets import *
from data import Data, FEMInverseProblem
from quadrature import MonteCarlo, QuasiMonteCarlo
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
		return diff.reshape((len(locations),1))/(2*self.ip.variance)

	def likelihood(self, locations):
		return np.exp(-self.potential(locations))

	def compute_norm_const(self, num_qmc_pts = 10000):

		def integrand(locations):
			return self.likelihood(locations) * self.prior_density(locations)

		num_true_inputs = len(self.ip.locations.T)
		self.norm_const = QuasiMonteCarlo.compute_integral(integrand, num_qmc_pts, num_true_inputs)

	def density(self, locations):
		if self.norm_const is None:
			print("Computing normalisation constant on N = 10000 pts...", end = "")
			self.compute_norm_const(10000)	
			print("done!")
		return self.likelihood(locations)/self.norm_const * self.prior_density(locations)




class ApproximatePosterior(Posterior):

	def __init__(self, posterior, gp):
		Posterior.__init__(self, posterior.ip, posterior.prior_density)
		self.gp = gp
		self.posterior = posterior

	def potential2(self, locations):
		diff = np.zeros(len(locations))
		evaluate = self.gp.mean_fct.evaluate(locations)
		diff = self.ip.observations  - evaluate
		normdiff = np.sum(np.abs(diff)**2,axis=-1)
		return normdiff.reshape((len(locations),1))/(2*self.ip.variance)

	"""
	num_observations is output dimension of forward model
	"""
	def makedata(self, locations, approximand, num_observations = 1):
		ip = self.posterior.ip
		observations = approximand(locations)
		self.approx_data = Data(locations, observations, 0.0)
		self.gp = ConditionedGaussianProcess(self.gp, self.approx_data)

	def approximate_forwardmap(self, pointset, num_observations = 1):
		assert(self.gp.is_conditioned == False), "Approximation already in use! Make new posterior"
		self.makedata(pointset, self.ip.forward_map)
		self.potential = self.potential2

	def potential3(self, locations):
		return self.gp.mean_fct.evaluate(locations).reshape((len(locations),1))

	def approximate_potential(self, pointset):
		assert(self.gp.is_conditioned == False), "Approximation already in use! Make new posterior"
		self.makedata(pointset, self.posterior.potential)
		self.potential = self.potential3


	def likelihood2(self, locations):
		meanfct =  self.gp.mean_fct.evaluate(locations).reshape((len(locations),1))
		return np.where(meanfct > 0, meanfct, 0)

	def approximate_likelihood(self, pointset):
		assert(self.gp.is_conditioned == False), "Approximation already in use! Make new posterior"
		self.makedata(pointset, self.posterior.likelihood)
		self.likelihood = self.likelihood2


class SampleApproximatePosterior(Posterior):

	def __init__(self, posterior, gp):
		Posterior.__init__(self, posterior.ip, posterior.prior_density)
		self.gp = gp
		self.posterior = posterior

	def potential2(self, locations):
		diff = np.zeros(len(locations))
		evaluate = self.gp.sample(locations).reshape((len(locations), 1))
		diff = self.ip.observations * np.ones(evaluate.shape) - evaluate
		normdiff = np.sum(np.abs(diff)**2,axis=-1)
		return normdiff.reshape((len(locations),1))/(2*self.ip.variance)

	"""
	num_observations is output dimension of forward model
	"""
	def makedata(self, locations, approximand, num_observations = 1):
		ip = self.posterior.ip
		observations = approximand(locations)
		self.approx_data = Data(locations, observations, 0.0)
		self.gp = ConditionedGaussianProcess(self.gp, self.approx_data)

	def approximate_forwardmap(self, pointset, num_observations = 1):
		assert(self.gp.is_conditioned == False), "Approximation already in use! Make new posterior"
		self.makedata(pointset, self.ip.forward_map)
		self.potential = self.potential2

	def potential3(self, locations):
		return self.gp.sample(locations).reshape((len(locations),1))

	def approximate_potential(self, pointset):
		assert(self.gp.is_conditioned == False), "Approximation already in use! Make new posterior"
		self.makedata(pointset, self.posterior.potential)
		self.potential = self.potential3


	def likelihood2(self, locations):
		samples =  self.gp.sample(locations).reshape((len(locations),1))
		return np.where(samples > 0, samples, 0)

	def approximate_likelihood(self, pointset):
		assert(self.gp.is_conditioned == False), "Approximation already in use! Make new posterior"
		self.makedata(pointset, self.posterior.likelihood)
		self.likelihood = self.likelihood2



class MarginalApproximatePosterior(SampleApproximatePosterior):

	def __init__(self, posterior, gp, num_mc_pts_vGN = 100):
		SampleApproximatePosterior.__init__(self, posterior, gp)
		self.num_mc_pts_vGN = num_mc_pts_vGN

	def potential2(self, locations):
		diff = np.zeros(len(locations))
		evaluate = self.gp.sample_many(locations, num_samps = self.num_mc_pts_vGN)#.reshape((len(locations),1))
		diff = self.ip.observations * np.ones(evaluate.shape) - evaluate
		normdiff = np.abs(diff)**2
		return normdiff/(2*self.ip.variance)

	"""
	num_observations is output dimension of forward model
	"""
	def makedata(self, locations, approximand, num_observations = 1):
		ip = self.posterior.ip
		observations = approximand(locations)
		self.approx_data = Data(locations, observations, 0.0)
		self.gp = ConditionedGaussianProcess(self.gp, self.approx_data)

	def approximate_forwardmap(self, pointset, num_observations = 1):
		assert(self.gp.is_conditioned == False), "Approximation already in use! Make new posterior"
		self.makedata(pointset, self.ip.forward_map)
		self.potential = self.potential2

	def potential3(self, locations):
		return self.gp.sample_many(locations, num_samps = self.num_mc_pts_vGN)#.reshape((len(locations),1))

	def approximate_potential(self, pointset):
		assert(self.gp.is_conditioned == False), "Approximation already in use! Make new posterior"
		self.makedata(pointset, self.posterior.potential)
		self.potential = self.potential3

	def approximate_likelihood(self, pointset):
		print("No marginal likelihood approximations")

	def compute_norm_const(self, num_qmc_pts = 10000):

		def integrand(locations):
			return self.likelihood(locations) * self.prior_density(locations)

		num_true_inputs = len(self.ip.locations.T)
		qmc = QuasiMonteCarlo.compute_integral(integrand, num_qmc_pts, num_true_inputs)
		self.norm_const = np.sum(qmc, axis = 1)/self.num_mc_pts_vGN
		


	def density(self, locations):
		if self.norm_const is None:
			print("Computing normalisation constant on N = 10000 pts...", end = "")
			self.compute_norm_const(10000)	
			print("done!")
		return (np.sum(self.likelihood(locations), axis = 1)).reshape((len(locations), 1))/(self.num_mc_pts_vGN * self.norm_const) * self.prior_density(locations)


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












