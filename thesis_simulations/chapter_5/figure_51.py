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
from bayesopt import *

from scipydirect import minimize as scdrmin
from scipy.optimize import minimize
from scipy.stats import norm
np.random.seed(1)

def pgfplotprint(A, B, title):
    print()
    print(title)
    assert(len(A) == len(B)), "lengths do not align"
    for i in range(len(A)):
        print("(", A[i,0], ",", B[i,0], ")")
    print()


np.random.seed(2)

def minimiser(pt):
    return 1/(pt + 1) * np.sin(10*pt)
#    return -np.sin(10*pt) - 2*(pt)**2

mesh = Mesh1d.construct(3)
ip = InverseProblem(mesh, minimiser, 0.)

zm = ZeroMean()
mc = MaternCov(2)
gp = GaussianProcess(zm, mc)
cgp = ConditionedGaussianProcess(gp, ip)
bop = BayesOpt(cgp, acq = "PI", thresh = 0.01)



bop.augment()
bop.augment()

cgpv2 = GPVisual(bop.cond_gp, num_pts = 250)
cgpv2.addplot_mean()
cgpv2.addplot_truth()
cgpv2.addplot_deviation()
cgpv2.addplot_observations()
vals = np.ones((len(cgpv2.mesh), 1))
for i in range(len(cgpv2.mesh)):
    vals[i, 0] = bop.acq_fct(cgpv2.mesh[i,:])
plt.plot(cgpv2.mesh, vals, '-', linewidth = 2, label = "acquisition fct")
plt.legend()
plt.show()
pgfplotprint(cgpv2.mesh, cgpv2.mean_vec, "Mean vec: ")
pgfplotprint(cgpv2.mesh, cgpv2.pos_dev, "Pos dev: ")
pgfplotprint(cgpv2.mesh, cgpv2.neg_dev, "Neg dev: ")
pgfplotprint(cgpv2.mesh, cgpv2.truth, "True Fct: ")
pgfplotprint(cgpv2.mesh, vals, "Acquisition fct: ")
pgfplotprint(cgpv2.gp.data.locations, cgpv2.gp.data.observations, "Observations: ")


sys.exit()

cgpv2 = GPVisual(bop.cond_gp, num_pts = 400)
cgpv2.addplot_mean()
cgpv2.addplot_truth()
cgpv2.addplot_deviation()
cgpv2.addplot_observations()
vals = np.ones((len(cgpv2.mesh), 1))
for i in range(len(cgpv2.mesh)):
    vals[i, 0] = 1*bop.acq_fct(cgpv2.mesh[i,:])
plt.plot(cgpv2.mesh, vals, '-', linewidth = 2, label = "acquisition fct")
plt.legend()
plt.show()
am = np.argmax(bop.cond_gp.mean_fct.evaluate(bop.mesh))

print("argmax:", bop.mesh[am])

pgfplotprint(cgpv2.mesh, vals, "acq fct:")

