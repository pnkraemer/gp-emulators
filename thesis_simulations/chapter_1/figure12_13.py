import sys
sys.path.insert(0, "../../modules")

from posterior import *
from data import *
from pointsets import *
from gaussianprocesses import *
from data import *
from prior import *
from quadrature import *


np.random.seed(1)

toy_ip = ToyInverseProblem1d(variance = 1e-1)
true_param = toy_ip.locations[0,0]

def priordens(locations):
    return Prior.gaussian1d_mix(locations, mean1 = 0.5 - np.abs(true_param - 0.5), variance1 = 0.05, mean2 = 0.5 + np.abs(true_param - 0.5), variance2 = 0.05)

posterior = Posterior(toy_ip, priordens)
posterior.compute_norm_const(100000)


num_plotpts = 250
locations = Mesh1d.construct(num_plotpts)



prior_density = posterior.prior_density(locations)
posterior_density = posterior.density(locations)


def cond_mean_integrand(locations):
    return locations[:,0] * posterior.density(locations)

cond_mean = MonteCarlo.compute_integral(cond_mean_integrand, 100000, 1)
cond_mean = np.array([[cond_mean]])

mapindex = np.argmax(posterior_density)


import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
plt.plot(locations, posterior_density, label ="posterior")
plt.plot(locations, prior_density, label ="prior")
plt.vlines(toy_ip.locations, 0, posterior.density(toy_ip.locations), label = "True input value")
plt.vlines(cond_mean, 0, posterior.density(cond_mean), linestyle = "dotted", label = "cond_mean")
plt.vlines(locations[mapindex,0], 0, posterior_density[mapindex], linestyle = "--", label = "map")
plt.legend()
plt.show()

print("\nPrior density:")
for i in range(num_plotpts):
    print("(", locations[i,0], ",", prior_density[i], ")")

print("\nPosterior density:")
for i in range(num_plotpts):
    print("(", locations[i,0], ",", posterior_density[i], ")")

print("\nTrue input/output:")
print("(", toy_ip.locations[0,0], ",", posterior.density(toy_ip.locations)[0], ")")

print("\nCond mean:\n", cond_mean[0,0])
print("\nMAP:\n", locations[mapindex, 0])
