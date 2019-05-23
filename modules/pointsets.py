"""
NAME: pointsets.py

AUTHOR: NK
"""

import numpy as np
from scipy.stats import norm


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

# stolen from https://laszukdawid.com/2017/02/04/halton-sequence-in-python/
class Halton():

    @staticmethod
    def construct_withzero(num_pts, dim):
        
        def nextPrime():
            
            def isPrime(num):
                for i in range(2,int(num**0.5)+1):
                    if(num % i)==0: return False
                return True
        
            prime = 3
            while(1):
                if isPrime(prime):
                    yield prime
                prime += 2

        def vanDerCorput(n, base=2):
            vdc, denom = 0, 1
            while n:
                denom *= base
                n, remainder = divmod(n, base)
                vdc += remainder/float(denom)
            return vdc

        seq = []
        primeGen = nextPrime()
        next(primeGen)
        for d in range(dim):
            base = next(primeGen)
            seq.append([vanDerCorput(i, base) for i in range(num_pts)])
        return np.array(seq).T

    """
    Construct Halton pointset, ignoring the first pt (0,0)
    """
    @staticmethod
    def construct(num_pts, dim):
        num_pts = num_pts + 1
        def nextPrime():
            
            def isPrime(num):
                for i in range(2,int(num**0.5)+1):
                    if(num % i)==0: return False
                return True
        
            prime = 3
            while(1):
                if isPrime(prime):
                    yield prime
                prime += 2

        def vanDerCorput(n, base=2):
            vdc, denom = 0, 1
            while n:
                denom *= base
                n, remainder = divmod(n, base)
                vdc += remainder/float(denom)
            return vdc

        seq = []
        primeGen = nextPrime()
        next(primeGen)
        for d in range(dim):
            base = next(primeGen)
            seq.append([vanDerCorput(i, base) for i in range(num_pts)])
        pts = np.array(seq).T
        return pts[1::, :]


"""
Lattice rules, see Frances Kuo's website for generating vectors
"""
class FibonacciSphere():

    # path is a string to the .txt file
    @staticmethod
    def construct(num_pts, rand_shift = True):
        rnd = 1.0
        if rand_shift:
            rnd = np.random.rand() * num_pts
        points = []
        offset = 2.0/num_pts
        increment = np.pi * (3.0 - np.sqrt(5.0));
        for idx in range(num_pts):
            y = ((idx * offset) - 1) + (offset / 2);
            r = np.sqrt(1 - y**2)
            phi = ((idx + rnd) % num_pts) * increment
            x = np.cos(phi) * r
            z = np.sin(phi) * r
            points.append([x,y,z])
        return np.array(points)


"""
Lattice rules, see Frances Kuo's website for generating vectors
"""
class Lattice():

    # path is a string to the .txt file
    @staticmethod
#    def construct(num_pts, dim, path = 'vectors/lattice-39102-1024-1048576.3600.txt', rand_shift = True):
    def construct(num_pts, dim, path = '/home/kraemer/Programme/gp-emulators/modules/vectors/lattice-39102-1024-1048576.3600.txt', rand_shift = True):

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


"""
MCMC: Metropolis-Hastings
"""
class MetropolisHastings():
    
    @staticmethod
    def sample1d(num_samps, prop_width, init_state = 0.0, density = norm.pdf):
        samples = np.zeros((num_samps, 1))
        curr_samp = init_state
        idx = 0
        samples[idx, 0] = curr_samp
        idx = idx + 1
        
        while idx < num_samps:
            proposal = curr_samp + prop_width * np.random.randn()
            acc_prob = density(proposal)/density(curr_samp)
            ratio = np.random.rand()
            if acc_prob < ratio:
                samples[idx, 0] = curr_samp
            else:
                samples[idx, 0] = proposal
                curr_samp = proposal
            idx = idx + 1
            
        return samples

    @staticmethod
    def sample1d_with_proposals(num_samps, prop_width, init_state = 0.0, density = norm.pdf):
        samples = np.zeros((num_samps, 1))
        proposals = np.zeros((num_samps, 1))
        accepted = np.zeros(num_samps)
        curr_samp = init_state
        idx = 0
        samples[idx, 0] = curr_samp
        proposals[idx, 0] = curr_samp
        accepted[idx] = 1
        idx = idx + 1
        
        while idx < num_samps:
            proposal = curr_samp + prop_width * np.random.randn()
            proposals[idx, 0] = proposal
            acc_prob = density(proposal)/density(curr_samp)
            ratio = np.random.rand()
            if acc_prob < ratio:
                samples[idx, 0] = curr_samp
            else:
                samples[idx, 0] = proposal
                curr_samp = proposal
                accepted[idx] = 1
            idx = idx + 1
            
        return samples, proposals, accepted


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
