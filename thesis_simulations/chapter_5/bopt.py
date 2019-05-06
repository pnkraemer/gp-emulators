import numpy as np
import sys
sys.path.insert(0, "../../modules")

from covariances import *
from means import *
from posterior import *
from data import *
from pointsets import *
from gaussianprocesses import *
from prior import *
from gpvisual import *
from scipydirect import minimize as scdrmin
from scipy.optimize import minimize
from scipy.stats import norm
np.random.seed(1)

"""
1D Bayesian Optimisation
"""
class BayesOpt():

	def __init__(self, cgp, acq = 'PI', thresh = 0.0, minimiser = "LBFGS", num_restarts = 10):

		assert(cgp.data.is_ip == True), 'Please initialise with an inverse problem'
		assert(cgp.is_conditioned == True), "Please initialise with a conditioned GP"
		self.acq = acq
		if self.acq == 'UCB':
			self.acq_fct = self.UCB
			self.thresh = None
		elif self.acq == 'PI':
			self.acq_fct = self.PI
			self.thresh = thresh
		elif self.acq == 'EI':
			self.acq_fct = self.EI
			self.thresh = thresh
		else:
			print("No acquisition function specified")
			assert(1==0), "No acquisition function specified"
		self.minimiser = minimiser
		self.cond_gp = cgp 
		self.mesh = self.cond_gp.data.locations
		self.last_argmax = self.mesh[np.argmax(self.cond_gp.mean_fct.evaluate(self.mesh))].reshape((1,1))
		self.num_restarts = num_restarts

	"""
	ptset is in (n,) format, fct returns negative upper confidence bound
	(reason: use minimization function)
	"""
	def PI(self, ptset):
		ptset = ptset.reshape((len(ptset), 1))
		
		last_max = np.max(self.cond_gp.mean_fct.evaluate(self.cond_gp.data.locations))

		m = self.cond_gp.mean_fct.evaluate(ptset)
		impr = m - last_max - self.thresh

		s = self.cond_gp.cov_fct.evaluate(ptset, ptset)
		if s[0,0]>0.:
			Z = impr/np.sqrt(s[0,0])
			return norm.cdf(Z)
		else: 
			return 0.

	"""
	ptset is in (n,) format, fct returns negative upper confidence bound
	(reason: use minimization function)
	"""
	def EI(self, ptset):
		ptset = ptset.reshape((len(ptset), 1))
		
		last_max = np.max(self.cond_gp.mean_fct.evaluate(self.cond_gp.data.locations))

		m = self.cond_gp.mean_fct.evaluate(ptset)
		impr = m - last_max - self.thresh

		s = self.cond_gp.cov_fct.evaluate(ptset, ptset)
		if s[0,0]>0.:
			Z = impr/np.sqrt(s[0,0])
			return impr * norm.cdf(Z) + np.sqrt(s[0,0])*norm.pdf(Z)
		else: 
			return 0.

	"""
	ptset is in (n,) format, fct returns negative upper confidence bound
	(reason: use minimization function)
	"""
	def UCB(self, ptset, num_devs = 2):
		assert(self.thresh is None), "Threshold specified-not suitable for UCB"
		ptset = ptset.reshape((len(ptset), 1))
		m = self.cond_gp.mean_fct.evaluate(ptset)
		s = np.diagonal(self.cond_gp.cov_fct.evaluate(ptset, ptset))
		return m + num_devs*np.sqrt(np.abs(s).reshape((len(s), 1)))

	def maximize(self, tolerance):

		def objective(ptset):
			return -1 * self.acq_fct(ptset)

		if self.minimiser == "DIRECT":
			res = scdrmin(objective, bounds = [(0, 1), ])#, tol = tolerance)
			resx = res.x.reshape((1,1))
		elif self.minimiser == "LBFGS":
			minval = 1
			for i in range(self.num_restarts):
				x0 = np.random.rand(1)
				res = minimize(objective, x0, bounds = ((0, 1), ), tol = tolerance, method='L-BFGS-B')
				if res.fun < minval:
					minval = res.fun[0]
					resx = res.x.reshape((1,1))
		#res = minimize(self.acq_fct, x0, bounds = ((0, 1), ), tol = tolerance)
		#res = minimize(self.acq_fct, x0, method = 'Nelder-Mead', tol = tolerance)
		return resx

	def augment(self, tolerance = 1e-20):
		maximum = self.maximize(tolerance = tolerance)
		self.mesh = np.concatenate((self.mesh, maximum), axis = 0)
		new_data = InverseProblem(self.mesh, self.cond_gp.data.forward_map, self.cond_gp.data.variance)
		self.cond_gp = ConditionedGaussianProcess(self.cond_gp.prior, new_data)



np.random.seed(2)

def minimiser(pt):
	return 1/(pt + 1) * np.sin(10*pt)
#	return -np.sin(10*pt) - 2*(pt)**2

mesh = Mesh1d.construct(2)
ip = InverseProblem(mesh, minimiser, 0.)

zm = ZeroMean()
mc = MaternCov(2.5)
gp = GaussianProcess(zm, mc)
cgp = ConditionedGaussianProcess(gp, ip)
bop = BayesOpt(cgp, acq = "PI", thresh = 0.001)

for i in range(5):
	bop.augment()
	cgpv2 = GPVisual(bop.cond_gp, num_pts = 400)
	cgpv2.addplot_mean()
	cgpv2.addplot_truth()
	cgpv2.addplot_deviation()
	cgpv2.addplot_observations()
	vals = np.ones(len(cgpv2.mesh))
	for i in range(len(cgpv2.mesh)):
		vals[i] = 1*bop.acq_fct(cgpv2.mesh[i,:])
	plt.plot(cgpv2.mesh, vals, '-', linewidth = 2, label = "acquisition fct")
	plt.legend()
	plt.show()
am = np.argmax(bop.cond_gp.mean_fct.evaluate(bop.mesh))

print("argmax:", bop.mesh[am])


