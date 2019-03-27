
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
num_obs = 4
noise = 0.0
data = ToyGPData(num_obs, noise)
zero_mean = ZeroMean()
cov_fct = MaternCov(1.)
gp = GaussianProcess(zero_mean, cov_fct)
cgp = ConditionedGaussianProcess(gp, data)



num_pts = 150
gpv = NakedGPVisual(cgp, ctheme = "darkslategray", num_pts = num_pts)
#gpv.addanimation_samples()
#gpv.addplot_samples(20)
gpv.addplot_mean()
gpv.addplot_deviation()
gpv.addplot_observations()
#plt.legend()
gpv.fig.savefig("./figures/gpani2.png", dpi = 500)
#gpv.anim.save("./figures/animations/gp_posterior_green.mp4", fps = 3, dpi = 250, writer="ffmpeg")#, fps=30, extra_args=['-vcodec', 'libx264'])
plt.show()






