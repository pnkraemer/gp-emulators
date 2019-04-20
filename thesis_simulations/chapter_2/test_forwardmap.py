
import numpy as np
import sys


sys.path.insert(0, "../../modules")
from pointsets import PointSet, Random, Mesh1d
from data import Data, InverseProblem, FEMInverseProblem
from quadrature import MonteCarlo
from posterior import Posterior, ApproximatePosterior
from means import ZeroMean
from covariances import MaternCov
from gaussianprocesses import GaussianProcess, ConditionedGaussianProcess
from gpvisual import GPVisual




np.random.seed(1)













# class ApproximatePosteriorForwardMap(Posterior):

# 	def __init__(self, inverse_problem, cond_gp):
# 		Posterior.__init__(self, inverse_problem)
# 		self.cond_gp = cond_gp 

# 	def potential(self, locations):
# 		diff = np.zeros(len(locations))
# 		for i in range(len(locations)):
# 			evaluate = self.cond_gp.mean_fct.evaluate(np.array([locations[i,:]]))
# 			diff[i] = np.linalg.norm(self.ip.observations - evaluate)**2
# 		return diff

# class ApproximatePosteriorPotential(Posterior):

# 	def __init__(self, inverse_problem, cond_gp):
# 		Posterior.__init__(self, inverse_problem)
# 		self.cond_gp = cond_gp 

# 	def potential(self, locations):
# 		diff = np.zeros(len(locations))
# 		for i in range(len(locations)):
# 			evaluate = self.cond_gp.mean_fct.evaluate(np.array([locations[i,:]]))
# 			diff[i] = evaluate
# 		return diff


# class ApproximatePosteriorLikelihood(Posterior):

# 	def __init__(self, inverse_problem, cond_gp):
# 		Posterior.__init__(self, inverse_problem)
# 		self.cond_gp = cond_gp 

# 	def likelihood(self, locations):
# 		approx_potent = np.zeros(len(locations))
# 		for i in range(len(locations)):
# 			evaluate = self.cond_gp.mean_fct.evaluate(np.array([locations[i,:]]))
# 			approx_potent[i] = evaluate
# 		return approx_potent




# class ApproxData(InverseProblem):

# 	def __init__(self, pointset, num_observations, posterior):
# 		ip = posterior.ip
# 		self.locations = pointset.points
# 		observations = np.zeros((len(pointset.points), num_observations))
# 		for i in range(len(pointset.points)):
# 			observations[i,:] = ip.forward_map(pointset.points[i,:])
# 		self.observations = observations
# 		self.true_observations = self.observations
# 		self.variance = 0.
# 		self.forward_map = ip.forward_map


# class ApproxDataPotential(InverseProblem):

# 	def __init__(self, pointset, num_observations, posterior):
# 		self.locations = pointset.points
# 		observations = posterior.potential(self.locations)
# 		observations = np.zeros((len(pointset.points), num_observations))
# 		for i in range(len(pointset.points)):
# 			observations[i,:] = posterior.potential(np.array([pointset.points[i,:]]))
# 		self.observations = observations
# 		self.true_observations = self.observations
# 		self.variance = 0.
# 		self.forward_map = posterior.potential

# class ApproxDataLikelihood(InverseProblem):

# 	def __init__(self, pointset, num_observations, posterior):
# 		self.locations = pointset.points
# 		observations = posterior.potential(self.locations)
# 		observations = np.zeros((len(pointset.points), num_observations))
# 		for i in range(len(pointset.points)):
# 			observations[i,:] = posterior.likelihood(np.array([pointset.points[i,:]]))
# 		self.observations = observations
# 		self.true_observations = self.observations
# 		self.variance = 0.
# 		self.forward_map = posterior.potential







num_true_inputs = 1
eval_pts = np.array([[0.5]])
meshwidth = 1./32.
variance = 1e-4
fem_ip = FEMInverseProblem(num_true_inputs, eval_pts, meshwidth, variance)
#print(fem_ip.locations)
#print(fem_ip.observations)
#print(fem_ip.true_observations)
#print(fem_ip.variance)





mean_fct = ZeroMean()
cov_fct = MaternCov(2.5)
gp = GaussianProcess(mean_fct, cov_fct)


num_design_pts = 25
design_ptset = Mesh1d(num_design_pts)




posterior = Posterior(fem_ip)
approx_post = ApproximatePosterior(posterior, gp)

#print(approx_post.potential)
approx_post.approximate_likelihood(design_ptset)
#print(approx_post.potential)




#gp_data = ApproxDataPotential(design_ptset, 1, posterior)
#cond_gp = ConditionedGaussianProcess(gp, gp_data)
#print(gp_data.locations,, gp_data.observations)

import matplotlib.pyplot as plt 

# gp_v = GPVisual(approx_post.cond_gp)
# gp_v.addplot_mean()
# gp_v.addplot_deviation()
# gp_v.addplot_observations()
# plt.show()




#aposterior = ApproximatePosteriorPotential(fem_ip, cond_gp)
posterior.compute_norm_const(1000)
#aposterior.compute_norm_const(1000)
approx_post.compute_norm_const(1000)

#print("Z =", posterior.potential(eval_pts))
#print("aZ =", aposterior.potential(eval_pts))
print("error_Z =", posterior.norm_const)
print("error_apprxs =", approx_post.norm_const)
#print("error_Z2 =", aposterior.norm_const)






#print(posterior.density(np.array([[0.5]])))
#print("fine so far")

A = Mesh1d(500)
B = posterior.density(A.points)
C = approx_post.density(A.points)
#D = aposterior.density(A.points)

plt.style.use("ggplot")
plt.plot(A.points, B, linewidth = 2, label ="posterior")
plt.plot(A.points, C, linewidth = 4, alpha = 0.5, label = "approx")
#plt.plot(A.points, D,'o', color = "black", label = "aposterior")
plt.legend()
plt.show()






