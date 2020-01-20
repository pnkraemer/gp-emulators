"""
TODO:
Test Metropolis Hastings points
Test lattices better
"""

import sys
from gpemu.pointsets import *

import unittest

import matplotlib.pyplot as plt 


"""
Unittest for 'Random' Class
Subclasses are corner cases: num_pts==0, resp. dim == 0
"""
class TestPointset():

	# This test class is supposed to collect methods and
	# construct to not fail any tests!
	def setup_standard(self):
		self.num_pts = 100
		self.dim = 2

	def test_shape_of_array_standard(self):
		self.assertEqual(len(self.ptset.shape), 2) # ptset is always a matrix
		self.assertEqual(len(self.ptset), self.num_pts)
		self.assertEqual(len(self.ptset.T), self.dim)


	@unittest.expectedFailure
	def test_try_if_dim_can_be_zero(self):
		self.ptset = Random.construct(0,1)

	@unittest.expectedFailure
	def test_try_if_num_pts_can_be_zero(self):
		self.ptset = Random.construct(0,1)


class TestPointsetUnitSquare(TestPointset):

	def test_if_bbox_is_contained_in_unitcube(self):
		self.assertGreaterEqual(np.amin(self.ptset), 0.)
		self.assertLessEqual(np.amax(self.ptset), 1.)

class TestRandom(unittest.TestCase, TestPointsetUnitSquare):

	def setUp(self):
		self.setup_standard()
		self.ptset = Random.construct(self.num_pts, self.dim)

class TestMesh1d(unittest.TestCase, TestPointsetUnitSquare):

	def setUp(self):
		self.setup_standard()
		self.dim = 1
		self.ptset = Mesh1d.construct(self.num_pts)


class TestHaltonWithZero(unittest.TestCase, TestPointsetUnitSquare):

	def setUp(self):
		self.setup_standard()
		self.ptset = Halton.construct(self.num_pts, self.dim)

class TestHaltonWithoutZero(unittest.TestCase, TestPointsetUnitSquare):

	def setUp(self):
		self.setup_standard()
		self.ptset = Halton.construct_withoutzero(self.num_pts, self.dim)

class TestLattice(unittest.TestCase, TestPointsetUnitSquare):

	def setUp(self):
		self.setup_standard()
		self.ptset = Lattice.construct(self.num_pts, self.dim, rand_shift = False)




class TestPointsetSphere(TestPointset):

	def test_if_points_are_spherical(self):
		rowwise_norm = np.linalg.norm(self.ptset, axis = 1)
		discr = np.linalg.norm(rowwise_norm - np.ones(rowwise_norm.shape))
		self.assertLess(discr, 1e-15)


class TestFibonacciSphere(unittest.TestCase, TestPointsetSphere):

	def setUp(self):
		self.setup_standard()
		self.dim = 3
		self.ptset = FibonacciSphere.construct(self.num_pts, rand_shift = False)






if __name__ == "__main__":
	unittest.main()






# import sys
# sys.path.insert(0, "../")
# from pointsets import *

# num_pts = 10000
# dim = 1

# print("\nChecking pointsets:")
# """
# Check 1d
# """
# print("\td=1...", end="")
# mesh1d = Mesh1d.construct(num_pts)
# random = Random.construct(num_pts, dim)
# lattice = Lattice.construct(num_pts, dim)
# assert(mesh1d.shape==random.shape)
# assert(mesh1d.shape==lattice.shape)
# assert(random.shape==lattice.shape)
# assert(random.shape==(num_pts, dim))
# print("successful!")
# halton = Halton.construct(num_pts, dim)
# assert(halton.shape==(num_pts, dim))

# """
# Check 5d
# """
# print("\td=5...", end="")
# dim = 5
# random = Random.construct(num_pts, dim)
# lattice = Lattice.construct(num_pts, dim)
# assert(random.shape==lattice.shape)
# assert(random.shape==(num_pts, dim))
# print("successful!")

# print("\nCheck Halton...", end = "")
# halton = Halton.construct_withzero(200, 2)
# halton2 = Halton.construct(200, 2)
# assert(len(halton) == len(halton2))
# print("successful!")

