import sys
sys.path.insert(0, "../../modules")

from pointsets import *
import numpy as np
from scipy.stats import norm

num_samps = 200
init_state = 0.0

print("\nSamples for s = 0.1:")
prop_width = 0.1
mcmc_samples1 = MetropolisHastings.sample1d(num_samps, prop_width, init_state)
for i in range(num_samps):
    print("(", i, ",", mcmc_samples1[i, 0], ")")


print("\nSamples for s = 10.0:")
prop_width = 10.0
mcmc_samples2 = MetropolisHastings.sample1d(num_samps, prop_width, init_state)
for i in range(num_samps):
    print("(", i, ",", mcmc_samples2[i, 0], ")")


print("\nSamples for x0 = 0.5:")
prop_width = 1.0
init_state = 0.5
mcmc_samples3 = MetropolisHastings.sample1d(num_samps, prop_width, init_state)
for i in range(num_samps):
    print("(", i, ",", mcmc_samples3[i, 0], ")")


print("\nSamples for s = 0.1:")
init_state = 12.0
mcmc_samples4 = MetropolisHastings.sample1d(num_samps, prop_width, init_state)
for i in range(num_samps):
    print("(", i, ",", mcmc_samples4[i, 0], ")")







import matplotlib.pyplot as plt 
plt.style.use("fivethirtyeight")

plt.figure()


plt.plot(mcmc_samples1, ".", label ="s = 0.1")
plt.plot(mcmc_samples2, ".", label ="s = 10.0")
plt.legend()
plt.show()


plt.figure()
plt.plot(mcmc_samples3, ".", label ="x0 = 0.5")
plt.plot(mcmc_samples4, ".", label ="x0 = 12.0")
plt.legend()
plt.show()
