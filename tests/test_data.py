import sys
sys.path.insert(0, "../")
from pointsets import *
from data import *


def forward_map(pt):
	return np.sin(20*pt)

num_pts = 100
dim = 1



ptset = Lattice.construct(num_pts, dim)
observations = forward_map(ptset)
variance = 1e-7

dataset = Data(ptset, observations, variance)
ip = InverseProblem(ptset, forward_map, variance)
print("\nChecking discrepancy of datasets:")
discr_obs = np.linalg.norm(dataset.observations - ip.observations)/np.sqrt(num_pts)
print("\tRMSE of Observations: %.1e"%(discr_obs), "(~%.1e?)"%(np.sqrt(variance)))

discr_trueobs = np.linalg.norm(dataset.true_observations - ip.true_observations)/np.sqrt(num_pts)
print("\tRMSE of true observations: %.1e"%(discr_trueobs), "(=0?)"%(np.sqrt(variance)))

discr_loc = np.linalg.norm(dataset.locations - ip.locations)/np.sqrt(num_pts)
print("\tRMSE of locations: %.1e"%(discr_trueobs), "(=0?)"%(np.sqrt(variance)))

discr_var = np.linalg.norm(dataset.variance - ip.variance)
print("\tDiff of variance: %.1e"%(discr_var), "(=0?)"%(np.sqrt(variance)))


print("\nAllocating toy datasets:")
toy_ip = ToyInverseProblem1d()
fem_ip = FEMInverseProblem()
toy_gp = ToyGPData1d()













