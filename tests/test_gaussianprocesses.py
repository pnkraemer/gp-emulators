

from gpemu.pointsets import *
from gpemu.data import *
from gpemu.gaussianprocesses import *
from gpemu.means import *
from gpemu.covariances import *
from gpemu.data import *

zeromean = ZeroMean()
materncov = MaternCov()
gpdata = ToyGPData1d()


gp = GaussianProcess(zeromean, materncov)
sgp = StandardGP()
cgp = ConditionedGaussianProcess(gp, gpdata)

sample_loc = Random.construct(10, 1)
gp.sample(sample_loc)
sgp.sample(sample_loc)
cgp.sample(sample_loc)



print("\nAssemblies and checks were successful\n")