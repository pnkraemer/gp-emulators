import numpy as np
import sys
sys.path.insert(0, "../../modules")

from posterior import *
from data import *
from pointsets import *
from gaussianprocesses import *
from prior import *
from quadrature import *


np.random.seed(2)

def hellinger_dist(post1, post2):
	assert(post1.ip.locations == post2.ip.locations)
	num_true_coeff = len(post1.ip.locations)
	def sqrtdens(ptset):
		return (np.sqrt(post1.density(ptset)) - np.sqrt(post2.density(ptset)))**2

	return 0.5 * np.sqrt(QuasiMonteCarlo.compute_integral(sqrtdens, 10000, num_true_coeff))

num_coeff = 3
num_design_pts = 4
design_ptset = Lattice.construct(num_design_pts, num_coeff, rand_shift = True)
print("\nN =", num_design_pts, "\n")

# Set up the FEM inverse problem
num_eval_pts = 1
eval_pts = Random.construct(num_eval_pts, 1)
fem_ip = FEMInverseProblem(input_dim = num_coeff, eval_pts = eval_pts)
print("FEM-IP set up")

# Set up the posterior distribution
posterior = Posterior(fem_ip, Prior.uniform)
posterior.compute_norm_const(10000)
print("True posterior set up")

# Approximate posterior distribution
prior_mean = ZeroMean()
prior_cov = MaternCov(1.5)
gp_prior = GaussianProcess(prior_mean, prior_cov)
approx_post = ApproximatePosterior(posterior, gp_prior)
approx_post.approximate_potential(design_ptset)
approx_post.compute_norm_const(10000)
print("Approximate posterior set up")

helldist=hellinger_dist(posterior, approx_post)
print("\nHellinger distance:\n\td = %.1e\n"%helldist)


