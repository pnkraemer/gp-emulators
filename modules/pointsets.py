"""
NAME: pointsets.py

AUTHOR: NK
"""

import numpy as np


"""
Random pointset
"""
class Random():

    @staticmethod
    def construct(num_pts, dim):
        return np.random.rand(num_pts, dim)

"""
Mesh in 1d
"""
class Mesh1d():

    @staticmethod
    def construct(num_pts):
        points = np.zeros((num_pts, 1))
        points[:,0] = np.linspace(0,1,num_pts)
        return points
"""
Lattice rules, see Frances Kuo's website for generating vectors
"""
class Lattice():

    # path is a string to the .txt file
    @staticmethod
    def construct(num_pts, dim, path = '/Users/nicholaskramer/Documents/GitHub/gp-emulators/modules/vectors/lattice-39102-1024-1048576.3600.txt', rand_shift = True):

        def load_gen_vec(path, dim):
            gen_vec = np.loadtxt(path)
            return gen_vec[0:dim, 1]

        gen_vec = load_gen_vec(path, dim)
        lattice = np.zeros((num_pts, dim))

        if rand_shift == True:
            shift = np.random.rand(dim)
            for i in range(num_pts):
                lattice[i, :] = (1.0*gen_vec * i / num_pts  + shift)% 1.0
        else: 
            for i in range(num_pts):
                lattice[i, :] = (1.0*gen_vec * i / num_pts)% 1.0
        return lattice







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
