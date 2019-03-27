
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

# x = np.linspace(0,1,200)
# y = likelihood(x)
# plt.style.use("ggplot")
# plt.plot(x,y, '-', linewidth = 2)
# plt.title("Likelihood")
# plt.show()


ptset = Mesh1d(12)
observations = likelihood(ptset.points)
data = Data(ptset, observations, 0.)

mean = ZeroMean()
cov = MaternCov(2.0)
gp = GaussianProcess(mean, cov)
cond_gp = ConditionedGaussianProcess(gp, data)

print(np.argmax(data.observations), data.observations[np.argmax(data.observations)], max(data.observations))
max_idx = np.argmax(data.observations)

mesh = Random(1, 1)
mesh.points[0,0] = 0.5
def exp_impr(pt, gp):
	a = gp.mean_fct.evaluate(pt) - data.observations[max_idx]
	normalised = a / gp.cov_fct.evaluate(pt, pt)
	b = gp.cov_fct.evaluate(pt, pt)
	z1 = norm.pdf(normalised)
	z2 = norm.cdf(normalised)
	print(a.shape, b.shape, normalised.shape)
	return a * z1 + b * z2
print(exp_impr(mesh.points, cond_gp))
print(mesh.points)


gpvisual = GPVisual(cond_gp)
gpvisual.addplot_mean()
gpvisual.addplot_deviation()
gpvisual.addplot_observations()
plt.show()

















# Set up problem and Gaussian process approximation
#num_obs = 4
#noise = 0.
#data = ToyGPData(num_obs, noise)
#zero_mean = ZeroMean()
#cov_fct = MaternCov(1.)
#gp = GaussianProcess(zero_mean, cov_fct)
#cgp = ConditionedGaussianProcess(gp, data)



num_pts = 200
#gpv = GPVisual(cgp, ctheme = "darkgreen")
#gpv.addanimation_samples()
#gpv.addplot_mean()
#gpv.addplot_deviation()
#gpv.addplot_observations()
#plt.legend()
#gpv.fig.savefig("./figures/gpani.png", dpi = 250)
#gpv.anim.save("./figures/animations/gp_posterior_green.mp4", fps = 3, dpi = 250, writer="ffmpeg")#, fps=30, extra_args=['-vcodec', 'libx264'])
#plt.show()






