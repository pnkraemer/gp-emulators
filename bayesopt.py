
import numpy as np
import sys
sys.path.insert(0, "./modules")

from data import ToyInverseProblem, ToyGPData, InverseProblem
from means import ZeroMean
from covariances import MaternCov
from gaussianprocesses import GaussianProcess, ConditionedGaussianProcess
from gpvisual import GPVisual
from likelihood import Likelihood


# Set up Inverse Problem
ip = ToyInverseProblem(10)
li = Likelihood(ip)

gpdata = ToyGPData(10)
gpdata = InverseProblem(gpdata.locations, li.function, gpdata.variance)












mean_fct = ZeroMean()
cov_fct = MaternCov()
gp = GaussianProcess(mean_fct, cov_fct)

cond_gp = ConditionedGaussianProcess(gp, gpdata)



import matplotlib.pyplot as plt 
gpvis = GPVisual(cond_gp)
gpvis.addplot_deviation()
gpvis.addplot_mean()
gpvis.addplot_observations()
plt.show()