
from gpemu.quadrature import *
from gpemu.pointsets import *

np.random.seed(2)

num_pts = 100000
dim = 1

def integrand(pt):
	return 123 * pt**2


print("\nCheck MC approximations:")
nodes = Random.construct(num_pts, dim)
weights = np.ones(num_pts)/(1.0*num_pts)

quadr = Quadrature(nodes, weights)
integral = quadr.compute(integrand)
residual = np.linalg.norm(integral - 123. / 3.0)
print("\tResidual: %.1e"%(residual), "(roughly %.1e?)"%(1.0/np.sqrt(num_pts)))

direct_integral = Quadrature.compute_integral(integrand, nodes, weights)
direct_residual = np.linalg.norm(direct_integral - 123. / 3.0)
print("\tResidual: %.1e"%(direct_residual), "(roughly %.1e?)"%(1.0/np.sqrt(num_pts)))
print("\t\t(Difference: %.1e"%(np.linalg.norm(direct_integral-integral)), "(=0?))")


mc = MonteCarlo(num_pts, dim)
mc_int = mc.compute(integrand)
mc_res = np.linalg.norm(mc_int - 123. / 3.0)
print("\tResidual: %.1e"%(mc_res), "(roughly %.1e?)"%(1.0/np.sqrt(num_pts)))

mc_int2 = MonteCarlo.compute_integral(integrand, num_pts, dim)
mc_res2 = np.linalg.norm(mc_int2 - 123. / 3.0)
print("\tResidual: %.1e"%(mc_res2), "(roughly %.1e?)"%(1.0/np.sqrt(num_pts)))

print("\nCheck QMC Approximations:")
qmc_nodes = Lattice.construct(num_pts, dim)
qmc_weights = np.ones(num_pts)/(1.0*num_pts)

qmc_quadr = Quadrature(qmc_nodes, qmc_weights)
qmc_integral = qmc_quadr.compute(integrand)
qmc_residual = np.linalg.norm(qmc_integral - 123. / 3.0)
print("\tResidual: %.1e"%(qmc_residual), "(roughly %.1e?)"%(1.0/(1.0*num_pts)))

qmc_direct_integral = Quadrature.compute_integral(integrand, qmc_nodes, qmc_weights)
qmc_direct_residual = np.linalg.norm(qmc_direct_integral - 123. / 3.0)
print("\tResidual: %.1e"%(qmc_direct_residual), "(roughly %.1e?)"%(1.0/(1.0*num_pts)))
print("\t\t(Difference: %.1e"%(np.linalg.norm(qmc_direct_integral-qmc_integral)), "(=0?))")



print("\nAll seems good \n")

