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
        self.num_pts = num_pts
        self.dim = dim
        self.points = np.zeros((num_pts, dim))
        self.bbox = np.zeros((dim, 2))
        self.bbox[:,1] = np.ones(dim)

    def reset_bbox_unitsquare(self):
        dim = self.dim
        self.bbox = np.zeros((dim, 2))
        self.bbox[:,1] = np.ones(dim)

    def affine_transform(self, new_bbox):

        def transform_pt(idx, new_bbox):
            
            pt = self.points[idx]
            dim = self.dim
            bbox = self.bbox

            for i in range(dim):
                scale = (new_bbox[i,1] - new_bbox[i,0])/(bbox[i,1] - bbox[i,0])
                shift = new_bbox[i,0] - bbox[i,0]
                pt[i] = shift + scale * pt[i]
            return pt

        points = self.points
        num_pts = self.num_pts
        dim = self.dim
        for i in range(num_pts):
        	points[i,:] = transform_pt(i, new_bbox)
        self.points = points
        self.bbox = new_bbox

"""
Random pointset
"""
class Random(PointSet):

    def __init__(self, num_pts, dim):
        PointSet.__init__(self, num_pts, dim)

    def construct_pointset(self):
        num_pts = self.num_pts
        dim = self.dim
        bbox = self.bbox

        self.points = np.random.rand(num_pts, dim)
        new_bbox = self.bbox
        self.reset_bbox_unitsquare()
        self.affine_transform(new_bbox)


"""
Lattice rules, see Frances Kuo's website for generating vectors
"""
class Lattice(PointSet):

    def __init__(self, num_pts, dim, rand_shift = False):
        PointSet.__init__(self, num_pts, dim)
        self.rand_shift = rand_shift

    # path is a string to the .txt file
    def load_gen_vec(self, path):
        dim = self.dim
        gen_vec = np.loadtxt(path)
        Lattice.gen_vec = gen_vec[0:dim, 1]

    def construct_pointset(self):
        num_pts = self.num_pts
        dim = self.dim
        gen_vec = self.gen_vec
        rand_shift = self.rand_shift

        lattice = np.zeros((num_pts, dim))
        if rand_shift == True:
            shift = np.random.rand(dim)
            for i in range(num_pts):
                lattice[i, :] = (1.0*gen_vec * i / num_pts  + shift)% 1.0
        else: 
            for i in range(num_pts):
                lattice[i, :] = (1.0*gen_vec * i / num_pts)% 1.0
        self.points = lattice
        new_bbox = self.bbox
        self.reset_bbox_unitsquare()
        self.affine_transform(new_bbox)






# """
# Some testing
# """
# import matplotlib.pyplot as plt 

# np.random.seed(1)
# num_pts = 100
# dim = 2
# unit_random = Random(num_pts, dim, seed = 1)
# unit_random2 = Random(num_pts, dim)
# # unit_random.load_gen_vec('vectors/lattice-39102-1024-1048576.3600.txt')
# unit_random.construct_pointset()
# unit_random2.construct_pointset()

# plt.style.use("ggplot")
# plt.plot(unit_random.points[:,0], unit_random.points[:,1], 'o')
# plt.plot(unit_random2.points[:,0], unit_random2.points[:,1], 'o')
# plt.show()
