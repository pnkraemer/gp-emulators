import sys
sys.path.insert(0, "../modules")
from posterior import *
from data import *
from pointsets import *
from gaussianprocesses import *


num_pts = 100
dim = 1

ptset = Lattice.construct(num_pts, dim)
ip = ToyInverseProblem1d()

posterior = Posterior(ip)
posterior.potential(ptset)
posterior.likelihood(ptset)
posterior.compute_norm_const()
posterior.density(ptset)

gp = StandardGP()
approx_post_frwmp = ApproximatePosterior(posterior, gp)
approx_post_frwmp.approximate_forwardmap(ptset)
approx_post_frwmp.density(ptset)



approx_post_pttl = ApproximatePosterior(posterior, gp)
approx_post_pttl.approximate_potential(ptset)
approx_post_pttl.density(ptset)

approx_post_llh = ApproximatePosterior(posterior, gp)
approx_post_llh.approximate_likelihood(ptset)
approx_post_llh.density(ptset)




print("\nAll seems good\n")

