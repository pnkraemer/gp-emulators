"""
NAME: pointsets.py

PURPOSE: knowledge about pointset generation
"""

import numpy as np


"""
Base class for pointsets
"""
class PointSet:

    def __init__(self, num_pts, dim):
        self.points = np.zeros((num_pts, dim))

    def augment(self, point):
        self.points = np.vstack((self.points, point))

"""
Random pointset
"""
class Random(PointSet):

    def construct_pointset_random(self, num_pts, dim):
        self.points = np.random.rand(num_pts, dim)

    def __init__(self, num_pts, dim):
        PointSet.__init__(self, num_pts, dim)
        self.construct_pointset_random(num_pts, dim)

"""
Mesh in 1d
"""
class Mesh1d(PointSet):

    def construct_pointset_mesh1d(self, num_pts):
        self.points = np.zeros((num_pts, 1))
        self.points[:,0] = np.linspace(0,1,num_pts)

    def __init__(self, num_pts):
        PointSet.__init__(self, num_pts, 1)
        self.construct_pointset_mesh1d(num_pts)


"""
Lattice rules, see Frances Kuo's website for generating vectors
"""
class Lattice(PointSet):

    # path is a string to the .txt file
    def construct_pointset_lattice(self, path, rand_shift):

        def load_gen_vec(path, dim):
            gen_vec = np.loadtxt(path)
            return gen_vec[0:dim, 1]

        dim = len(self.points.T)
        num_pts = len(self.points)
        gen_vec = load_gen_vec(path, dim)
        lattice = np.zeros((num_pts, dim))

        if rand_shift == True:
            shift = np.random.rand(dim)
            for i in range(num_pts):
                lattice[i, :] = (1.0*gen_vec * i / num_pts  + shift)% 1.0
        else: 
            for i in range(num_pts):
                lattice[i, :] = (1.0*gen_vec * i / num_pts)% 1.0
        self.points = lattice

    def __init__(self, num_pts, dim, path = '/Users/nicholaskramer/Documents/GitHub/gp-emulators/modules/vectors/lattice-39102-1024-1048576.3600.txt', rand_shift = False):
        PointSet.__init__(self, num_pts, dim)
        self.construct_pointset_lattice(path, rand_shift)







# # """
# # Some testing
# # """
# import matplotlib.pyplot as plt 

# np.random.seed(1)
# num_pts = 1000
# dim = 2
# ptset = Lattice(num_pts, dim, rand_shift = False)
# print(ptset.points)
# ptset2 = Lattice(num_pts, dim, rand_shift = True, seed = 1)
# ptset3 = Lattice(num_pts, dim, rand_shift = True, seed = 2)
# # unit_random2 = Random(num_pts, dim)
# # # unit_random.load_gen_vec('vectors/lattice-39102-1024-1048576.3600.txt')
# # unit_random.construct_pointset()
# # unit_random2.construct_pointset()

# plt.style.use("ggplot")
# plt.plot(ptset.points[:,0], ptset.points[:,1], 'o', label = "No shift")
# plt.plot(ptset2.points[:,0], ptset2.points[:,1], 'o', label = "Shift, seed = 1")
# plt.plot(ptset3.points[:,0], ptset3.points[:,1], 'o', label = "Shift, seed = 2")
# # plt.plot(unit_random2.points[:,0], unit_random2.points[:,1], 'o')
# plt.legend()
# plt.show()
