import numpy as np
import sys
sys.path.insert(0, "../../modules")

from covariances import *
from gaussianprocesses import *
from pointsets import *
from data import *
from distances import *

np.random.seed(1)

def forward_map(pt):
	return np.sin(pt)

num_it = 11
print("\nnu = 1.5")
num_datapts = 1
for i in range(num_it):
	num_datapts = 2*num_datapts
	dataset = Mesh1d.construct(num_datapts)

	ip = InverseProblem(dataset, forward_map, variance = 0.)

	zm = ZeroMean()
	m1 = MaternCov(1.0)
	gp1 = GaussianProcess(zm, m1)
	cgp1 = ConditionedGaussianProcess(gp1, ip)
	rmse1 = RMSE.compute(cgp1.mean_fct.evaluate, forward_map)
	print("( %i , %.1e )" %(num_datapts, rmse1))


print("\nnu = 2.5")
num_datapts = 1
for i in range(num_it):
	num_datapts = 2*num_datapts
	dataset = Mesh1d.construct(num_datapts)

	ip = InverseProblem(dataset, forward_map, variance = 0.)
	zm = ZeroMean()

	m2 = MaternCov(2.0)
	gp2 = GaussianProcess(zm, m2)
	cgp2 = ConditionedGaussianProcess(gp2, ip)
	rmse2 = RMSE.compute(cgp2.mean_fct.evaluate, forward_map)
	print("( %i , %.1e )" %(num_datapts, rmse2))


print("\nnu = 3.0")
num_datapts = 1
for i in range(num_it):
	num_datapts = 2*num_datapts
	dataset = Mesh1d.construct(num_datapts)

	ip = InverseProblem(dataset, forward_map, variance = 0.)
	zm = ZeroMean()

	m3 = MaternCov(3.0)
	gp3 = GaussianProcess(zm, m3)
	cgp3 = ConditionedGaussianProcess(gp3, ip)
	rmse3 = RMSE.compute(cgp3.mean_fct.evaluate, forward_map)
	print("( %i , %.1e )" %(num_datapts, rmse3))


