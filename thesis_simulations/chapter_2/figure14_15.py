import sys
sys.path.insert(0, "../../modules")

from pointsets import *

from scipy.stats import norm

num_samps = 20000
prop_width = 3.0

mcmc_samples, proposals, accepted = MetropolisHastings.sample1d_with_proposals(num_samps, prop_width)

import matplotlib.pyplot as plt 
import numpy as np

np.random.seed(7)
plt.style.use("fivethirtyeight")
plt_locations = np.linspace(-6, 6, 250)
plt_values_gaussdens = norm.pdf(plt_locations)


histbars, bin_edges = np.histogram(mcmc_samples, bins = 35, density = True)
bin_edges = bin_edges[:-1] + np.diff(bin_edges)/2


print("\nGauss density values:")
for i in range(len(plt_locations)):
    print("(", plt_locations[i], ",", plt_values_gaussdens[i], ")")


print("\nHistogram values:")
for i in range(len(histbars)):
    print("(", bin_edges[i], ",", histbars[i], ")")

print("\nProposals, Samples, acceptance:")
for i in range(6):
    print("(", proposals[i,0], ",", mcmc_samples[i, 0], ",", accepted[i],  ")")














plt.figure()
plt.xlabel("Location")
plt.ylabel("Probability / Rel. Frequency")
plt.plot(plt_locations, plt_values_gaussdens, label ="Gaussian density")
plt.hist(mcmc_samples, bins = 50, density = 1, label ="MCMC samples")
plt.legend()
plt.grid(False)
plt.show()
