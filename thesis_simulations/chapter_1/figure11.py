import sys
sys.path.insert(0, "../../modules")

from posterior import *
from prior import *
from data import *
from pointsets import *
from gaussianprocesses import *
from data import *
np.random.seed(10)

toy_ip = ToyInverseProblem1d(variance = 1e-1)
true_param = toy_ip.locations[0,0]

posterior = Posterior(toy_ip, Prior.uniform)
posterior.compute_norm_const(100000)


num_plotpts = 250
locations = Mesh1d.construct(num_plotpts)

prior_density = posterior.prior_density(locations)
posterior_density = posterior.density(locations)

import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
plt.plot(locations, posterior_density, label ="posterior")
plt.plot(locations, prior_density, label ="prior")
plt.vlines(toy_ip.locations, 0, posterior.density(toy_ip.locations), label = "True input value")
plt.legend()
plt.show()

print("\nPrior input/output:")
for i in range(num_plotpts):
    print("(", locations[i,0], ",", prior_density[i], ")")

print("\nPosterior input/output:")
for i in range(num_plotpts):
    print("(", locations[i,0], ",", posterior_density[i], ")")

print("\nTrue input/output:")
print("(", toy_ip.locations[0,0], ",", posterior.density(toy_ip.locations)[0], ")")
