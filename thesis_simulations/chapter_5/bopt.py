import numpy as np
import sys
sys.path.insert(0, "../../modules")

from posterior import *
from data import *
from pointsets import *
from gaussianprocesses import *
from prior import *
from gpvisual import *
from scipydirect import minimize
from scipy.stats import norm
np.random.seed(1)

"""
1D Bayesian Optimisation
"""
class BayesOpt():

	def __init__(self, cgp, acq = 'PI', thresh = 0.):

		assert(cgp.data.is_ip == True), 'Please initialise with an inverse problem'
		assert(cgp.is_conditioned == True), "Please initialise with a conditioned GP"
		if acq == 'UCB':
			self.acq_fct = self.UCB
			self.thresh = None
		elif acq == 'PI':
			self.acq_fct = self.PI
			self.thresh = thresh
		else:
			print("No acquisition function specified")
			assert(1==0), "No acquisition function specified"
		self.cond_gp = cgp 
		self.mesh = self.cond_gp.data.locations
		self.last_argmax = self.mesh[np.argmax(self.cond_gp.mean_fct.evaluate(self.mesh))].reshape((1,1))

	"""
	ptset is in (n,) format, fct returns negative upper confidence bound
	(reason: use minimization function)
	"""
	def PI(self, ptset):
		ptset = ptset.reshape((len(ptset), 1))
		
		last_max = self.cond_gp.mean_fct.evaluate(self.last_argmax)

		m = self.cond_gp.mean_fct.evaluate(ptset)

		s = np.diagonal(self.cond_gp.cov_fct.evaluate(ptset, ptset))
		if s>np.finfo(float).eps:
			return norm.cdf((m - last_max - self.thresh)/s)
		else: 
			return 0

	"""
	ptset is in (n,) format, fct returns negative upper confidence bound
	(reason: use minimization function)
	"""
	def UCB(self, ptset):
		ptset = ptset.reshape((len(ptset), 1))
		m = self.cond_gp.mean_fct.evaluate(ptset)
		s = np.diagonal(self.cond_gp.cov_fct.evaluate(ptset, ptset))
		return -1 * (m + 2*np.sqrt(np.abs(s).reshape((len(s), 1))))

	def maximize(self, x0, tolerance):
		res = minimize(self.acq_fct, bounds = [(0, 1), ])#, tol = tolerance)
		#res = minimize(self.acq_fct, x0, bounds = ((0, 1), ), tol = tolerance)
		#res = minimize(self.acq_fct, x0, method = 'Nelder-Mead', tol = tolerance)
		return res.x.reshape((1,1))

	def augment(self, x0 = np.random.rand(1), tolerance = 1e-14):
		maximum = self.maximize(x0 = x0, tolerance = tolerance)
		self.mesh = np.concatenate((self.mesh, maximum), axis = 0)
		new_data = InverseProblem(self.mesh, self.cond_gp.data.forward_map, self.cond_gp.data.variance)
		self.cond_gp = ConditionedGaussianProcess(self.cond_gp.prior, new_data)
		if self.cond_gp.mean_fct.evaluate(maximum.reshape((1,1))) > self.cond_gp.mean_fct.evaluate(self.last_argmax.reshape((1,1))):
			self.last_argmax = maximum



np.random.seed(2)

def minimiser(pt):
	return 1/(pt + 1) * np.sin(10*pt)

mesh = Mesh1d.construct(5)
ip = InverseProblem(mesh, minimiser, 0.)

gp = StandardGP()
cgp = ConditionedGaussianProcess(gp, ip)
bop = BayesOpt(cgp, thresh = 1e-10)

for i in range(3):
	print("Last argmax:", bop.last_argmax)
	bop.augment()
print(bop.cond_gp.data.locations, "(0.148294)")
cgpv2 = GPVisual(bop.cond_gp)
cgpv2.addplot_truth()
cgpv2.addplot_mean()
cgpv2.addplot_deviation()
cgpv2.addplot_observations()
plt.show()


