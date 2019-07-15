import sys
sys.path.insert(0, "../")
from covariances import *
from pointsets import *

num_pts = 250
dim = 1

ptset = Random.construct(num_pts, dim)

print("\nChecking discrepancy of static methods:")
gauss_cov = GaussCov()
cov_mtrx1 = gauss_cov.evaluate(ptset, ptset)
cov_mtrx2 = GaussCov.fast_mtrx(ptset, ptset)
discrep_gauss = np.linalg.norm(cov_mtrx1 - cov_mtrx2)
print("\t...Gauss: %.1e"%(discrep_gauss))

exp_cov = ExpCov()
cov_mtrx1 = exp_cov.evaluate(ptset, ptset)
cov_mtrx2 = ExpCov.fast_mtrx(ptset, ptset)
discrep_exp = np.linalg.norm(cov_mtrx1 - cov_mtrx2)
print("\t...Exp: %.1e"%(discrep_exp))

matern_cov = MaternCov()
cov_mtrx1 = matern_cov.evaluate(ptset, ptset)
cov_mtrx2 = MaternCov.fast_mtrx(ptset, ptset)
discrep_matern = np.linalg.norm(cov_mtrx1 - cov_mtrx2)
print("\t...Matern: %.1e"%(discrep_matern))


print("\nCan they interpolate?")

def rhsfunct(coord):
	return coord**2

rhs_vec = rhsfunct(ptset)
eval_ptset = Random.construct(num_pts, dim)
rhs_eval = rhsfunct(eval_ptset)


gauss_mtrx = GaussCov.fast_mtrx(ptset, ptset)
coeff_gauss = np.linalg.solve(gauss_mtrx, rhs_vec)
eval_mtrx_gauss = GaussCov.fast_mtrx(eval_ptset, ptset)
val_gauss = eval_mtrx_gauss.dot(coeff_gauss)
rmse_gauss = np.linalg.norm(val_gauss - rhs_eval)/np.sqrt(num_pts)
print("\t...RMSE of Gauss: %.1e"%(rmse_gauss))

exp_mtrx = ExpCov.fast_mtrx(ptset, ptset)
coeff_exp = np.linalg.solve(exp_mtrx, rhs_vec)
eval_mtrx_exp = ExpCov.fast_mtrx(eval_ptset, ptset)
val_exp = eval_mtrx_exp.dot(coeff_exp)
rmse_exp = np.linalg.norm(val_exp - rhs_eval)/np.sqrt(num_pts)
print("\t...RMSE of Exp: %.1e"%(rmse_exp))

matern_mtrx = MaternCov.fast_mtrx(ptset, ptset)
coeff_matern = np.linalg.solve(matern_mtrx, rhs_vec)
eval_mtrx_matern = MaternCov.fast_mtrx(eval_ptset, ptset)
val_matern = eval_mtrx_matern.dot(coeff_matern)
rmse_matern = np.linalg.norm(val_matern - rhs_eval)/np.sqrt(num_pts)
print("\t...RMSE of Matern: %.1e"%(rmse_matern))
















