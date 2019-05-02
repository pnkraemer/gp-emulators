"""
NAME: distances.py

PURPOSE: Distance functions (Hellinger, RMSE, ...) to measure errors
"""
import numpy as np
from pointsets import *
from quadrature import *

class Distance():
	pass


"""
RMSE.random takes functions which take (only) pointsets as input and have 1d output   
"""
class RMSE(Distance):

	@staticmethod
	def compute(truth, function, num_evals = 9999, eval_dim = 1):
		eval_ptset = Random.construct(num_evals, eval_dim)
		return np.linalg.norm(truth(eval_ptset) - function(eval_ptset), ord = None) / np.sqrt(num_evals)


"""
Hellinger distance between posterior (distributions)
"""
class Hellinger(Distance):

	@staticmethod
	def compute(post1, post2, num_qmc_pts = 10000):
		assert(np.linalg.norm(post1.ip.locations - post2.ip.locations) == 0)
		num_true_coeff = len(post1.ip.locations.T)

		def sqrtdens(ptset):
			return (np.sqrt(post1.density(ptset)) - np.sqrt(post2.density(ptset)))**2 * post1.prior_density(ptset)

		return np.sqrt(0.5 * QuasiMonteCarlo.compute_integral(sqrtdens, num_qmc_pts, num_true_coeff))



