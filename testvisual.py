
import numpy as np
import sys
sys.path.insert(0, "./modules")
from pointsets import PointSet, Random, Mesh1d
from covariances import GaussCov, ExpCov, MaternCov
from means import ZeroMean
from data import ToyGPData
from gaussianprocesses import GaussianProcess, ConditionedGaussianProcess
from gpvisual import GPVisual, NakedGPVisual



from matplotlib import pyplot as plt
from matplotlib import animation

np.random.seed(1)





# Set up problem and Gaussian process approximation
num_obs = 6
noise = 0.
data = ToyGPData(num_obs, noise)
zero_mean = ZeroMean()
cov_fct = MaternCov(6.5)
gp = GaussianProcess(zero_mean, cov_fct)
cgp = ConditionedGaussianProcess(gp, data)



num_pts = 200
gpv = GPVisual(gp)
gpv.addanimation_samples()
#gpv.addplot_mean()
gpv.addplot_deviation()
gpv.addplot_observations()
#gpv.addplot_errorbar()
plt.legend()

plt.show()

#animation_gp(gp, '../animations/animation_prior.mp4', num_pts)
#animation_gp(cGP, '../animations/animation_posterior.mp4', num_pts)




