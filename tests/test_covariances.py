"""
TODO:

make test where it is checked that matern with nu = 0.5 equals exponential and 
that matern with nu = 99999 kinda equals gaussian
"""


import unittest
import numpy as np
import sys
sys.path.insert(0, "../modules/")
from covariances import *
from pointsets import *

np.random.seed(15051994)

"""
Base class for testing covariance functions
Not part of unittest.TestCase because we only collect a few methods and dont want this to count as a test already
"""
class TestCovariance():

	"""
	Set up a standard test case
	"""
	def setup_standard(self):
		self.num_pts_small = 50
		self.num_pts_large = 250
		self.dim = 1
		self.ptset_small = Lattice.construct(self.num_pts_small, self.dim, rand_shift = False)
		self.ptset_large = Halton.construct(self.num_pts_large, self.dim)
		self.shift = 11.1111
		self.corr_length = 1.0
		self.num_evalpts = 5000

	"""
	Set up one edge test case
	"""
	def setup_lowpoints_highdimension(self):
		self.num_pts_small = 50
		self.num_pts_large = 250
		self.dim = 1
		self.ptset_small = Lattice.construct(self.num_pts_small, self.dim, rand_shift = False)
		self.ptset_large = Halton.construct(self.num_pts_large, self.dim)
		self.shift = 11.1111
		self.corr_length = 1.0
		self.num_evalpts = 10000

	def test_classmethod_staticmethod_same_matrix(self):
		discr = np.linalg.norm(self.cov_mtrx_static - self.cov_mtrx_class) 	# discrepancy
		self.assertLess(discr, 1e-15)

	def test_did_a_shift_happen(self):
		discr = np.linalg.norm(self.cov_mtrx_symmetric[0,0] - (self.expected_diagonal_element + self.shift))	# discrepancy
		self.assertLess(discr, 1e-15)


class TestCovariancePosDef(TestCovariance):									# pos. def. as opposed to conditionally pos. def.

	def test_shapes_of_cov_mtrx(self):
		self.assertEqual(len(self.cov_mtrx_static.shape), 2)				# 2-array (matrix)
		self.assertEqual(len(self.cov_mtrx_static), self.num_pts_small)
		self.assertEqual(len(self.cov_mtrx_static.T), self.num_pts_large)
		discr = np.linalg.norm(self.cov_mtrx_static - self.cov_mtrx_class) 	# discrepancy
		self.assertLess(discr, 1e-15)

	def test_can_it_interpolate_pos_def(self):
		int_mtrx = self.cov_fct.evaluate(self.ptset_large, self.ptset_large, shift = 0.0)
		rhs = self.ptset_large[:,0]			# interpolate f(x,y) = x
		sol = np.linalg.solve(int_mtrx, rhs)
		evalptset = Random.construct(self.num_evalpts, self.dim)
		evalmtrx = self.cov_fct.evaluate(evalptset, self.ptset_large)
		approx_sol = evalmtrx.dot(sol)
		rmse = np.linalg.norm(approx_sol - evalptset[:,0])/np.sqrt(self.num_evalpts)
		self.assertLess(rmse, self.expected_interpolation_error)


"""
Unittests begin here
"""
class TestGaussCov(unittest.TestCase, TestCovariancePosDef):

	def setUp(self):
		TestCovariance.setup_standard(self)
		self.cov_mtrx_static = GaussCov.fast_mtrx(self.ptset_small, self.ptset_large, corr_length = self.corr_length, shift = self.shift)
		self.cov_fct = GaussCov(corr_length = self.corr_length)
		self.cov_mtrx_class = self.cov_fct.evaluate(self.ptset_small, self.ptset_large, shift = self.shift)
		self.cov_mtrx_symmetric = GaussCov.fast_mtrx(self.ptset_large, self.ptset_large, corr_length = self.corr_length, shift = self.shift)
		self.expected_interpolation_error = 1e-07   # expected 3.329473984604855e-08 for seed 15051994
		self.expected_diagonal_element = 1.0

class TestGaussCovLowPtsHighDim(TestGaussCov):

	def setUp(self):
		TestCovariance.setup_lowpoints_highdimension(self)
		self.cov_mtrx_static = GaussCov.fast_mtrx(self.ptset_small, self.ptset_large, corr_length = self.corr_length, shift = self.shift)
		self.cov_fct = GaussCov(corr_length = self.corr_length)
		self.cov_mtrx_class = self.cov_fct.evaluate(self.ptset_small, self.ptset_large, shift = self.shift)
		self.cov_mtrx_symmetric = GaussCov.fast_mtrx(self.ptset_large, self.ptset_large, corr_length = self.corr_length, shift = self.shift)
		self.expected_interpolation_error = 10.0
		self.expected_diagonal_element = 1.0


class TestExpCov(unittest.TestCase, TestCovariancePosDef):

	def setUp(self):
		TestCovariance.setup_standard(self)
		self.cov_mtrx_static = ExpCov.fast_mtrx(self.ptset_small, self.ptset_large, corr_length = self.corr_length, shift = self.shift)
		self.cov_fct = ExpCov(corr_length = self.corr_length)
		self.cov_mtrx_class = self.cov_fct.evaluate(self.ptset_small, self.ptset_large, shift = self.shift)
		self.cov_mtrx_symmetric = ExpCov.fast_mtrx(self.ptset_large, self.ptset_large, corr_length = self.corr_length, shift = self.shift)
		self.expected_interpolation_error = 1e-03   # expected 0.0006038222047165582 for seed 15051994
		self.expected_diagonal_element = 1.0



class TestExpCovLowPtsHighDim(TestExpCov):

	def setUp(self):
		TestCovariance.setup_lowpoints_highdimension(self)
		self.cov_mtrx_static = ExpCov.fast_mtrx(self.ptset_small, 
			self.ptset_large, corr_length = self.corr_length, shift = self.shift)
		self.cov_fct = ExpCov(corr_length = self.corr_length)
		self.cov_mtrx_class = self.cov_fct.evaluate(self.ptset_small, 
			self.ptset_large, shift = self.shift)
		self.cov_mtrx_symmetric = ExpCov.fast_mtrx(self.ptset_large, 
			self.ptset_large, corr_length = self.corr_length, shift = self.shift)
		self.expected_interpolation_error = 10.0
		self.expected_diagonal_element = 1.0


"""
Wrapper for function to setup matern, no unittest
"""
class TestMaternCov(TestCovariancePosDef):

	def setup_matern(self, regularity):
		self.smoothness = regularity
		self.cov_mtrx_static = MaternCov.fast_mtrx(self.ptset_small, self.ptset_large, smoothness = self.smoothness, corr_length = self.corr_length, shift = self.shift)
		self.cov_fct = MaternCov(corr_length = self.corr_length, smoothness = self.smoothness)
		self.cov_mtrx_class = self.cov_fct.evaluate(self.ptset_small, 
			self.ptset_large, shift = self.shift)
		self.cov_mtrx_symmetric = MaternCov.fast_mtrx(self.ptset_large, 
			self.ptset_large, corr_length = self.corr_length, shift = self.shift)
		self.expected_diagonal_element = 1.0


class TestMaternCovHighReg(unittest.TestCase, TestMaternCov):

	def setUp(self):
		TestCovariance.setup_standard(self)
		self.setup_matern(7.5)
		self.expected_interpolation_error = 1e-07	# expected 1.9264961507228897e-08 for seed (15051994)

class TestMaternCovHighRegLowPtsHighDim(TestMaternCov):

	def setUp(self):
		TestCovariance.setup_lowpoints_highdimension(self)
		self.setup_matern(7.5)
		self.expected_interpolation_error = 10.0


class TestMaternCovLowReg(unittest.TestCase, TestMaternCov):

	def setUp(self):
		TestCovariance.setup_standard(self)
		self.setup_matern(0.25)
		self.expected_interpolation_error = 1e-2 # expected 0.0021523540730878613 for seed 15051994


class TestMaternCovLowRegLowPtsHighDim(TestMaternCov):

	def setUp(self):
		TestCovariance.setup_lowpoints_highdimension(self)
		self.setup_matern(0.25)
		self.expected_interpolation_error = 10.0


class TestCovarianceCondPosDef(TestCovariance):

	def test_shapes_of_cov_mtrx(self):
		self.assertEqual(len(self.cov_mtrx_static.shape), 2)				# matrix is a 2d-array (matrix)
		self.assertEqual(len(self.cov_mtrx_static), self.num_pts_small + self.dim + 1)		# correct number of rows
		self.assertEqual(len(self.cov_mtrx_static.T), self.num_pts_large + self.dim + 1)	# correct number of cols
		discr = np.linalg.norm(self.cov_mtrx_static - self.cov_mtrx_class) 	# discrepancy
		self.assertLess(discr, 1e-15)										# both matrices are same up to machine precision

	def test_can_it_interpolate_cond_pos_def(self):
		int_mtrx = self.cov_fct.evaluate(self.ptset_large, self.ptset_large, shift = 0.0)
		rhs = np.zeros(len(int_mtrx))
		rhs[:len(self.ptset_large)] = self.ptset_large[:,0]			# interpolate f(x,y) = x
		sol = np.linalg.solve(int_mtrx, rhs)
		evalptset = Random.construct(self.num_evalpts, self.dim)
		evalmtrx = self.cov_fct.evaluate(evalptset, self.ptset_large)
		approx_sol = evalmtrx.dot(sol)
		rmse = np.linalg.norm(approx_sol[:len(evalptset)] - evalptset[:,0])/np.sqrt(self.num_evalpts)
		self.assertLess(rmse, self.expected_interpolation_error)


class TestTpsCov(unittest.TestCase, TestCovarianceCondPosDef):

	def setUp(self):
		TestCovariance.setup_standard(self)
		self.shift = 0.0	# WORKAROUND / thin plate spline implementation does not handle shifts yet
		self.cov_mtrx_static = TpsCov.fast_mtrx(self.ptset_small, self.ptset_large)
		self.cov_fct = TpsCov()
		self.cov_mtrx_class = self.cov_fct.evaluate(self.ptset_small, self.ptset_large)
		self.cov_mtrx_symmetric = TpsCov.fast_mtrx(self.ptset_large, self.ptset_large)
		self.expected_interpolation_error = 1e-14	# expected 9.38251012645541e-16 for seed 15051994
		self.expected_diagonal_element = 0.0


class TestTpsCovLowPtsHighDim(TestTpsCov):

	def setUp(self):
		TestCovariance.setup_lowpoints_highdimension(self)
		self.shift = 0.0	# WORKAROUND / thin plate spline implementation does not handle shifts yet
		self.cov_mtrx_static = TpsCov.fast_mtrx(self.ptset_small, self.ptset_large)
		self.cov_fct = TpsCov()
		self.cov_mtrx_class = self.cov_fct.evaluate(self.ptset_small, self.ptset_large)
		self.cov_mtrx_symmetric = TpsCov.fast_mtrx(self.ptset_large, self.ptset_large)
		self.expected_interpolation_error = 10.0	# expected 9.38251012645541e-16 for seed 15051994
		self.expected_diagonal_element = 0.0


class TestTpsSphereCov(unittest.TestCase, TestCovarianceCondPosDef):

	def setup_standard_sphere(self):
		self.num_pts_small = 50
		self.num_pts_large = 250
		self.dim = 3
		self.ptset_small = FibonacciSphere.construct(self.num_pts_small, rand_shift = False)
		self.ptset_large = FibonacciSphere.construct(self.num_pts_large, rand_shift = False)
		self.shift = 11.1111
		self.corr_length = 1.0
		self.num_evalpts = 5000


	def setUp(self):
		self.setup_standard_sphere()
		self.shift = 0.0	# WORKAROUND / thin plate spline implementation does not handle shifts yet
		self.cov_mtrx_static = TpsSphereCov.fast_mtrx(self.ptset_small, self.ptset_large)
		self.cov_fct = TpsSphereCov()
		self.cov_mtrx_class = self.cov_fct.evaluate(self.ptset_small, self.ptset_large)
		self.cov_mtrx_symmetric = TpsSphereCov.fast_mtrx(self.ptset_large, self.ptset_large)
		self.expected_diagonal_element = 0.0
		self.expected_interpolation_error = 1e-14	# expected 1.2692951072607123e-15 for seed 15051994


	# def test_compare_evaluate_and_fast_mtrx_zero_shift(self):
	# 	shift = 0.0
	# 	cov_fct = GaussCov(self.corr_length)
		
	# 	# Different pointsets
	# 	mtrx1 = cov_fct.evaluate(self.ptset_small, self.ptset_large, shift)
	# 	mtrx2 = GaussCov.fast_mtrx(self.ptset_small, self.ptset_large, self.corr_length, shift)
	# 	discr = np.linalg.norm(mtrx1 - mtrx2)
	# 	self.assertLess(discr, 1e-14)

	# 	# Same pointsets
	# 	mtrx1 = cov_fct.evaluate(self.ptset_small, self.ptset_small, shift)
	# 	mtrx2 = GaussCov.fast_mtrx(self.ptset_small, self.ptset_small, self.corr_length, shift)
	# 	discr = np.linalg.norm(mtrx1 - mtrx2)
	# 	self.assertLess(discr, 1e-14)

	# def test_compare_evaluate_and_fast_mtrx_nonzero_shift(self):
	# 	shift = 9.9999
	# 	cov_fct = GaussCov(self.corr_length)
		
	# 	# Different pointsets
	# 	mtrx1 = cov_fct.evaluate(self.ptset_small, self.ptset_large, shift)
	# 	mtrx2 = GaussCov.fast_mtrx(self.ptset_small, self.ptset_large, self.corr_length, shift)
	# 	discr = np.linalg.norm(mtrx1 - mtrx2)
	# 	self.assertLess(discr, 1e-14)

	# 	# Same pointsets
	# 	mtrx1 = cov_fct.evaluate(self.ptset_small, self.ptset_small, shift)
	# 	mtrx2 = GaussCov.fast_mtrx(self.ptset_small, self.ptset_small, self.corr_length, shift)
	# 	discr = np.linalg.norm(mtrx1 - mtrx2)
	# 	self.assertLess(discr, 1e-14)

	# 	# Did a shift happen?
	# 	self.assertGreaterEqual(mtrx1[0,0], shift)

	# def test_compare_evaluate_and_fast_mtrx_no_input(self):
	# 	shift = 999.9999
	# 	cov_fct = GaussCov()
		
	# 	# Different pointsets
	# 	mtrx1 = cov_fct.evaluate(self.ptset_small, self.ptset_large)
	# 	mtrx2 = GaussCov.fast_mtrx(self.ptset_small, self.ptset_large)
	# 	discr = np.linalg.norm(mtrx1 - mtrx2)
	# 	self.assertLess(discr, 1e-14)

	# 	# Same pointsets
	# 	mtrx1 = cov_fct.evaluate(self.ptset_small, self.ptset_small)
	# 	mtrx2 = GaussCov.fast_mtrx(self.ptset_small, self.ptset_small)
	# 	discr = np.linalg.norm(mtrx1 - mtrx2)
	# 	self.assertLess(discr, 1e-14)

	# 	# Hopefully no shift and corr.length = 1.0?
	# 	self.assertLess(mtrx1[0,0], shift)
	# 	self.assertEqual(cov_fct.corr_length, 1.0)



if __name__ == "__main__":

	unittest.main()















# import sys
# sys.path.insert(0, "../")
# from covariances import *
# from pointsets import *

# num_pts = 250
# dim = 1

# ptset = Random.construct(num_pts, dim)

# print("\nChecking discrepancy of static methods:")
# gauss_cov = GaussCov()
# cov_mtrx1 = gauss_cov.evaluate(ptset, ptset)
# cov_mtrx2 = GaussCov.fast_mtrx(ptset, ptset)
# discrep_gauss = np.linalg.norm(cov_mtrx1 - cov_mtrx2)
# print("\t...Gauss: %.1e"%(discrep_gauss))

# exp_cov = ExpCov()
# cov_mtrx1 = exp_cov.evaluate(ptset, ptset)
# cov_mtrx2 = ExpCov.fast_mtrx(ptset, ptset)
# discrep_exp = np.linalg.norm(cov_mtrx1 - cov_mtrx2)
# print("\t...Exp: %.1e"%(discrep_exp))

# matern_cov = MaternCov()
# cov_mtrx1 = matern_cov.evaluate(ptset, ptset)
# cov_mtrx2 = MaternCov.fast_mtrx(ptset, ptset)
# discrep_matern = np.linalg.norm(cov_mtrx1 - cov_mtrx2)
# print("\t...Matern: %.1e"%(discrep_matern))


# print("\nCan they interpolate?")

# def rhsfunct(coord):
# 	return coord**2

# rhs_vec = rhsfunct(ptset)
# eval_ptset = Random.construct(num_pts, dim)
# rhs_eval = rhsfunct(eval_ptset)


# gauss_mtrx = GaussCov.fast_mtrx(ptset, ptset)
# coeff_gauss = np.linalg.solve(gauss_mtrx, rhs_vec)
# eval_mtrx_gauss = GaussCov.fast_mtrx(eval_ptset, ptset)
# val_gauss = eval_mtrx_gauss.dot(coeff_gauss)
# rmse_gauss = np.linalg.norm(val_gauss - rhs_eval)/np.sqrt(num_pts)
# print("\t...RMSE of Gauss: %.1e"%(rmse_gauss))

# exp_mtrx = ExpCov.fast_mtrx(ptset, ptset)
# coeff_exp = np.linalg.solve(exp_mtrx, rhs_vec)
# eval_mtrx_exp = ExpCov.fast_mtrx(eval_ptset, ptset)
# val_exp = eval_mtrx_exp.dot(coeff_exp)
# rmse_exp = np.linalg.norm(val_exp - rhs_eval)/np.sqrt(num_pts)
# print("\t...RMSE of Exp: %.1e"%(rmse_exp))

# matern_mtrx = MaternCov.fast_mtrx(ptset, ptset)
# coeff_matern = np.linalg.solve(matern_mtrx, rhs_vec)
# eval_mtrx_matern = MaternCov.fast_mtrx(eval_ptset, ptset)
# val_matern = eval_mtrx_matern.dot(coeff_matern)
# rmse_matern = np.linalg.norm(val_matern - rhs_eval)/np.sqrt(num_pts)
# print("\t...RMSE of Matern: %.1e"%(rmse_matern))
















