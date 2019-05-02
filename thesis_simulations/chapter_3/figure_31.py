import numpy as np
import sys
sys.path.insert(0, "../../modules")

from posterior import *
from data import *
from pointsets import *
from gaussianprocesses import *
from prior import *
from quadrature import *
from distances import Hellinger

np.random.seed(2)

num_it = 10
num_coeff = 1
num_qmc_pts = 10000
num_design_pts = 1
s = 1.5
prior_mean = ZeroMean()
prior_cov = MaternCov(s)
gp_prior = GaussianProcess(prior_mean, prior_cov)

# Set up the FEM inverse problem
num_eval_pts = 1
eval_pts = Random.construct(num_eval_pts, 1)
fem_ip = FEMInverseProblem(input_dim = num_coeff, eval_pts = eval_pts)
print("\nnum_qmc =", num_qmc_pts)
print("\ns =", s)
print("\nAppr. of forward map:")
for i in range(num_it):
	num_design_pts = 2*num_design_pts

	# Set up the posterior distribution
	posterior = Posterior(fem_ip, Prior.uniform)
	posterior.compute_norm_const(num_qmc_pts)

	# Approximate posterior distribution
	design_ptset = Lattice.construct(num_design_pts, num_coeff, rand_shift = True)
	approx_post = ApproximatePosterior(posterior, gp_prior)
	approx_post.approximate_forwardmap(design_ptset)
	approx_post.compute_norm_const(num_qmc_pts)

	# Approximation error
	helldist = Hellinger.compute(posterior, approx_post, num_qmc_pts)
	print("( %d , %.1e )" %(num_design_pts, helldist))

print("\nAppr. of potential:")
num_design_pts = 1
for i in range(num_it):
	num_design_pts = 2*num_design_pts

	# Set up the posterior distribution
	posterior = Posterior(fem_ip, Prior.uniform)
	posterior.compute_norm_const(num_qmc_pts)

	# Approximate posterior distribution
	design_ptset = Lattice.construct(num_design_pts, num_coeff, rand_shift = True)
	approx_post = ApproximatePosterior(posterior, gp_prior)
	approx_post.approximate_potential(design_ptset)
	approx_post.compute_norm_const(num_qmc_pts)

	# Approximation error
	helldist = Hellinger.compute(posterior, approx_post, num_qmc_pts)
	print("( %d , %.1e )" %(num_design_pts, helldist))


s = 3.5
prior_mean = ZeroMean()
prior_cov = MaternCov(s)
gp_prior = GaussianProcess(prior_mean, prior_cov)

# Set up the FEM inverse problem
num_eval_pts = 1
eval_pts = Random.construct(num_eval_pts, 1)
fem_ip = FEMInverseProblem(input_dim = num_coeff, eval_pts = eval_pts)
num_design_pts = 1

print("\n\ns =", s)
print("\nAppr. of forward map:")
for i in range(num_it):
	num_design_pts = 2*num_design_pts

	# Set up the posterior distribution
	posterior = Posterior(fem_ip, Prior.uniform)
	posterior.compute_norm_const(num_qmc_pts)

	# Approximate posterior distribution
	design_ptset = Lattice.construct(num_design_pts, num_coeff, rand_shift = True)
	approx_post = ApproximatePosterior(posterior, gp_prior)
	approx_post.approximate_forwardmap(design_ptset)
	approx_post.compute_norm_const(num_qmc_pts)

	# Approximation error
	helldist = Hellinger.compute(posterior, approx_post, num_qmc_pts)
	print("( %d , %.1e )" %(num_design_pts, helldist))

print("\nAppr. of potential:")
num_design_pts = 1
for i in range(num_it):
	num_design_pts = 2*num_design_pts

	# Set up the posterior distribution
	posterior = Posterior(fem_ip, Prior.uniform)
	posterior.compute_norm_const(num_qmc_pts)

	# Approximate posterior distribution
	design_ptset = Lattice.construct(num_design_pts, num_coeff, rand_shift = True)
	approx_post = ApproximatePosterior(posterior, gp_prior)
	approx_post.approximate_potential(design_ptset)
	approx_post.compute_norm_const(num_qmc_pts)

	# Approximation error
	helldist = Hellinger.compute(posterior, approx_post, num_qmc_pts)
	print("( %d , %.1e )" %(num_design_pts, helldist))


