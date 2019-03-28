
import numpy as np
import sys
sys.path.insert(0, "./modules")
#from pointsets import PointSet, Random, Mesh1d
#from covariances import GaussCov, ExpCov, MaternCov
#from means import ZeroMean
#from data import ToyGPData
#from gaussianprocesses import GaussianProcess, ConditionedGaussianProcess
#from gpvisual import GPVisual, NakedGPVisual





from pointsets import Random, Mesh1d
from data import Data 
from gaussianprocesses import GaussianProcess, ConditionedGaussianProcess
from means import ZeroMean
from covariances import MaternCov
from gpvisual import GPVisual

from matplotlib import pyplot as plt
from matplotlib import animation
from scipy.stats import norm
np.random.seed(1)


def forwardmap(x):
	return np.sin(5*x)

def likelihood(x, y = 0.51):
	return np.exp(-(y - forwardmap(x))**2/0.2)


def exp_impr(pt, gp):
	a = gp.mean_fct.evaluate(pt) - data.observations[max_idx]
	normalised = a / gp.cov_fct.evaluate(pt, pt)
	b = gp.cov_fct.evaluate(pt, pt)
	z1 = norm.pdf(normalised)
	z2 = norm.cdf(normalised)
	if a > 0:
		return a + b * z1 - np.abs(a) * z2
	else:
		return b * z1 - np.abs(a) * z2


# x = np.linspace(0,1,200)
# y = likelihood(x)
# plt.style.use("ggplot")
# plt.plot(x,y, '-', linewidth = 2)
# plt.title("Likelihood")
# plt.show()


ptset = Mesh1d(3)
mesh = Mesh1d(100)
ei = np.zeros(mesh.num_pts)
ei[0] = 1

for i in range(10):
	while np.max(ei) > 1e-14:
		#print(np.max(ei))
		#print(ptset.points)
		observations = likelihood(ptset.points)
		data = Data(ptset, observations, 0.)

		mean = ZeroMean()
		cov = MaternCov(1.5)
		gp = GaussianProcess(mean, cov)
		cond_gp = ConditionedGaussianProcess(gp, data)

		#print(np.argmax(data.observations), data.observations[np.argmax(data.observations)], max(data.observations))
		max_idx = np.argmax(data.observations)

		mesh = Mesh1d(100)

		ei = np.zeros(mesh.num_pts)
		for i in range(mesh.num_pts):
			ei[i] = exp_impr(np.array([mesh.points[i]]), cond_gp)
			#print("E-Imp =", exp_impr(np.array([mesh.points[i]]), cond_gp), i)
			#print("E-Im: ", exp_impr(mesh.points[i], cond_gp))
		#print(exp_impr(mesh.points, cond_gp))
		idx = np.argmax(ei)
		gpvisual = GPVisual(cond_gp)
		gpvisual.addplot_mean()
		gpvisual.addplot_deviation()
		gpvisual.addplot_observations()
		plt.legend()
		xvalues = np.linspace(0,1,100)
		plt.plot(xvalues, likelihood(xvalues), '--', linewidth = 2)
		plt.plot(mesh.points[idx], cond_gp.mean_fct.evaluate(np.array([mesh.points[idx]])), 'X', label = "Max. Exp. Impr")
		plt.show()
		ptset.augment(np.array([mesh.points[idx]]))

observations = likelihood(ptset.points)
data = Data(ptset, observations, 0.)

mean = ZeroMean()
cov = MaternCov(1.25)
gp = GaussianProcess(mean, cov)
cond_gp = ConditionedGaussianProcess(gp, data)

gpvisual = GPVisual(cond_gp)
gpvisual.addplot_mean()
gpvisual.addplot_deviation()
gpvisual.addplot_observations()
#plt.plot(mesh.points[idx], cond_gp.mean_fct.evaluate(np.array([mesh.points[idx]])), 'X', label = "Max. Exp. Impr")
plt.title("FINAL APPROXIMATION")
xvalues = np.linspace(0,1,100)
plt.plot(xvalues, likelihood(xvalues), '--', linewidth = 2, label ="True likelihood")
plt.legend()
plt.show()






