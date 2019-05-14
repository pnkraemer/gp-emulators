import sys
sys.path.insert(0, "../../modules")

from gpvisual import *
from gaussianprocesses import *
from data import *


np.random.seed(1)
num_pltpts = 250


gp = StandardGP()
gp_data = ToyGPData1d(num_pts = 3)
cgp = ConditionedGaussianProcess(gp, gp_data)


gpvis_prior = GPVisual(gp, num_pts = num_pltpts)
gpvis_prior.addplot_mean()
gpvis_prior.addplot_deviation(3)
plt.legend()
plt.show()


print("\nPrior mean:")
#for i in range(num_pltpts):
#    print("(", gpvis_prior.mesh[i,0], ",", gpvis_prior.mean_vec[i,0], ")")

print("\nPrior Pos Dev:")
#for i in range(num_pltpts):
#    print("(", gpvis_prior.mesh[i,0], ",", gpvis_prior.pos_dev[i,0], ")")

print("\nPrior Neg Dev:")
#for i in range(num_pltpts):
#    print("(", gpvis_prior.mesh[i,0], ",", gpvis_prior.neg_dev[i,0], ")")




gpvis = GPVisual(cgp, num_pts = num_pltpts)
gpvis.addplot_mean()
gpvis.addplot_truth()
gpvis.addplot_deviation()
gpvis.addplot_observations()
plt.legend()
plt.show()

print("\nPredictive mean:")
for i in range(num_pltpts):
    print("(", gpvis.mesh[i,0], ",", gpvis.mean_vec[i,0], ")")

print("\nPredictive Pos Dev:")
for i in range(num_pltpts):
    print("(", gpvis.mesh[i,0], ",", gpvis.pos_dev[i,0], ")")

print("\nPredictive Neg Dev:")
for i in range(num_pltpts):
    print("(", gpvis.mesh[i,0], ",", gpvis.neg_dev[i,0], ")")


print("\nTrue function:")
for i in range(num_pltpts):
    print("(", gpvis.mesh[i,0], ",", gpvis.gp.data.forward_map(gpvis.mesh)[i,0], ")")


print("\nObservations:")
for i in range(3):
    print("(", cgp.data.locations[i,0], ",", cgp.data.observations[i,0], ")")






