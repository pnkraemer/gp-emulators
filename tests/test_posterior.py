import sys
sys.path.insert(0, "../modules")
from posterior import *
from data import *
from pointsets import *
from gaussianprocesses import *
from prior import *

num_pts = 100
dim = 1

ptset = Lattice.construct(num_pts, dim)
#ip = FEMInverseProblem()
ip = ToyInverseProblem1d()

posterior = Posterior(ip, Prior.uniform)
posterior.potential(ptset)
posterior.likelihood(ptset)
posterior.compute_norm_const()
posterior.density(ptset)

gp = StandardGP()
approx_post_frwmp = ApproximatePosterior(posterior, gp)
approx_post_frwmp.approximate_forwardmap(ptset)
approx_post_frwmp.density(ptset)
approx_post_frwmp.density(ptset)


gp = StandardGP()
approx_post_pttl = ApproximatePosterior(posterior, gp)
approx_post_pttl.approximate_potential(ptset)
approx_post_pttl.density(ptset)

gp = StandardGP()
approx_post_llh = ApproximatePosterior(posterior, gp)
approx_post_llh.approximate_likelihood(ptset)
approx_post_llh.density(ptset)

gp = StandardGP()
approx_post_frwmp = SampleApproximatePosterior(posterior, gp)
approx_post_frwmp.approximate_forwardmap(ptset)
approx_post_frwmp.compute_norm_const(1000)
approx_post_frwmp.density(ptset)



gp = StandardGP()
approx_post_pttl = SampleApproximatePosterior(posterior, gp)
approx_post_pttl.approximate_potential(ptset)
approx_post_pttl.compute_norm_const(1000)
approx_post_pttl.density(ptset)

gp = StandardGP()
approx_post_llh = SampleApproximatePosterior(posterior, gp)
approx_post_llh.approximate_likelihood(ptset)
approx_post_llh.compute_norm_const(1000)
approx_post_llh.density(ptset)

print("\n1d fine\n")





print("\nCheck plots\n")

import matplotlib.pyplot as plt 
plt.style.use("fivethirtyeight")

pts = Mesh1d.construct(200)
gp = StandardGP()
posterior.compute_norm_const(1000)
val2 = posterior.density(pts)



approx_post = ApproximatePosterior(posterior, gp)
approx_post.compute_norm_const(1000)
approx_post.approximate_forwardmap(ptset)
approx_post.density(ptset)
val1 = approx_post.density(pts)
plt.figure()
plt.plot(pts, val1, '-', label = "mean_approx forwardmap", alpha = 0.5)
plt.plot(pts, val2, '-', label = "truth", alpha = 0.5)
plt.legend()
plt.show()

approx_post = ApproximatePosterior(posterior, gp)
approx_post.compute_norm_const(1000)
approx_post.approximate_potential(ptset)
approx_post.density(ptset)
val1 = approx_post.density(pts)
plt.figure()
plt.plot(pts, val1, '-', label = "mean_approx potential", alpha = 0.5)
plt.plot(pts, val2, '-', label = "truth", alpha = 0.5)
plt.legend()
plt.show()

approx_post = ApproximatePosterior(posterior, gp)
approx_post.compute_norm_const(1000)
approx_post.approximate_likelihood(ptset)
approx_post.density(ptset)
val1 = approx_post.density(pts)
plt.figure()
plt.plot(pts, val1, '-', label = "mean_approx likelihood", alpha = 0.5)
plt.plot(pts, val2, '-', label = "truth", alpha = 0.5)
plt.legend()
plt.show()

approx_post = SampleApproximatePosterior(posterior, gp)
approx_post.compute_norm_const(1000)
approx_post.approximate_forwardmap(ptset)
approx_post.density(ptset)
val1 = approx_post.density(pts)
plt.figure()
plt.plot(pts, val1, '-', label = "sample_approx forwardmap", alpha = 0.5)
plt.plot(pts, val2, '-', label = "truth", alpha = 0.5)
plt.legend()
plt.show()

approx_post = SampleApproximatePosterior(posterior, gp)
approx_post.compute_norm_const(1000)
approx_post.approximate_potential(ptset)
approx_post.density(ptset)
val1 = approx_post.density(pts)
plt.figure()
plt.plot(pts, val1, '-', label = "sample_approx potential", alpha = 0.5)
plt.plot(pts, val2, '-', label = "truth", alpha = 0.5)
plt.legend()
plt.show()

approx_post = SampleApproximatePosterior(posterior, gp)
approx_post.compute_norm_const(1000)
approx_post.approximate_likelihood(ptset)
approx_post.density(ptset)
val1 = approx_post.density(pts)
plt.figure()
plt.plot(pts, val1, '-', label = "sample_approx likelihood", alpha = 0.5)
plt.plot(pts, val2, '-', label = "truth", alpha = 0.5)
plt.legend()
plt.show()


print("\nAll seems good\n")

