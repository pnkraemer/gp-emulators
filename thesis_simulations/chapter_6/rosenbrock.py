import numpy as np
import sys
sys.path.insert(0, "../../modules")

from covariances import *
from means import *
from posterior import *
from data import *
from pointsets import *
from gaussianprocesses import *
from prior import *
from gpvisual import *
from bayesopt import *

from scipydirect import minimize as scdrmin
from scipy.optimize import minimize
from scipy.stats import norm

np.random.seed(1)


def pgfplotprint(A, B, title):
    print()
    print(title)
    assert(len(A) == len(B)), "lengths do not align"
    for i in range(len(A)):
        print("(", A[i,0], ",", B[i,0], ")")
    print()





def cheapmaximiser(X, A = 1., B = 100.):
    x = 2*X[:,0]
    y = 2*X[:,1]
    a = (1 - x)**2
    b = (1 - y)**2
    return -1 * (a + b).reshape((len(X), 1))

"""
x in [-2, 2]; y in [-2,2]
"""
def rosenbrock_transform(X):
    x = 2*X[:,0] 
    y = 2*X[:,1] 
    return x,y

def rosenbrock(X, A = 1., B = 100.):
    x,y = rosenbrock_transform(X)
    a = (A - x)**2
    b = B*(y - x*x)**2
    return -1 * (a + b).reshape((len(X), 1))


'''
x in [-1.5, 4]; y in [-3,4]
'''
def mccormick_transform(X):
    x = 5.5*X[:,0] - 1.5
    y = 7*X[:,1] - 3
    x = x.reshape((len(x), 1))
    y = y.reshape((len(x), 1))
    Y = np.concatenate((x,y), axis = 1)
    return Y

def mccormick(X):
    Y = mccormick_transform(X)
    x = Y[:,0]
    y = Y[:,1]
    a = np.sin(x + y)
    b = (x-y)**2
    c = 2.5*y - 1.5*x + 1
    return -1*(a + b + c).reshape((len(X), 1))


num_halton = 9
mesh = Halton.construct(num_halton, 2)

zm = ZeroMean()
mc = MaternCov(2)
gp = GaussianProcess(zm, mc)
num_augm = 35








"""
cheap
"""
ip = InverseProblem(mesh, cheapmaximiser, 0.)
cgp = ConditionedGaussianProcess(gp, ip)
#bop = BayesOpt2d(cgp, acq = "UCB", minimiser = "DIRECT")
bop = BayesOpt2d(cgp, acq = "UCB", minimiser = "DIRECT")

for i in range(num_augm):
    bop.augment()

mshpts = bop.mesh[num_halton::]
evalpts = cheapmaximiser(mshpts)

x,y = rosenbrock_transform(mshpts)
x = x.reshape((len(x), 1))
y = y.reshape((len(x), 1))
Y = np.concatenate((x,y), axis = 1)
rosenmin = 0
pgfplotprint(np.arange(len(evalpts)).reshape((len(evalpts), 1)) + 1, np.abs(rosenmin - (-1)*evalpts), "cheapmax-UCB-DIRECT errors:")

sys.exit()



"""
MC CORMICK
"""
zm = ZeroMean()
mc = MaternCov(2)
gp = GaussianProcess(zm, mc)
mesh = Halton.construct(num_halton, 2)
ip = InverseProblem(mesh, mccormick, 0.)
cgp = ConditionedGaussianProcess(gp, ip)
#bop = BayesOpt2d(cgp, acq = "UCB", minimiser = "DIRECT")
bop = BayesOpt2d(cgp, acq = "UCB", minimiser = "DIRECT")

for i in range(num_augm):
    bop.augment()

mshpts = bop.mesh[num_halton::]
evalpts = mccormick(mshpts)

Y = mccormick_transform(mshpts)
mccormickmin = -1.9133
pgfplotprint(np.arange(len(evalpts)).reshape((len(evalpts), 1)) + 1, np.abs(mccormickmin - (-1)*evalpts), "mccormick-UCB-DIRECT errors:")











"""
ROSENBROCK
"""
ip = InverseProblem(mesh, rosenbrock, 0.)
cgp = ConditionedGaussianProcess(gp, ip)
#bop = BayesOpt2d(cgp, acq = "UCB", minimiser = "DIRECT")
bop = BayesOpt2d(cgp, acq = "UCB", minimiser = "DIRECT")

for i in range(num_augm):
    bop.augment()

mshpts = bop.mesh[num_halton::]
evalpts = rosenbrock(mshpts)

x,y = rosenbrock_transform(mshpts)
x = x.reshape((len(x), 1))
y = y.reshape((len(x), 1))
Y = np.concatenate((x,y), axis = 1)
rosenmin = 0
pgfplotprint(np.arange(len(evalpts)).reshape((len(evalpts), 1)) + 1, np.abs(rosenmin - (-1)*evalpts), "rosenbrock-UCB-DIRECT errors:")





































sys.exit()


np.random.seed(2)

def minimiser(pt):
    return 1/(pt[0] + 1) * np.sin(10*pt[1])
#    return -np.sin(10*pt) - 2*(pt)**2

mesh = Halton.construct(3, 2)
ip = InverseProblem(mesh, minimiser, 0.)

zm = ZeroMean()
mc = MaternCov(2)
gp = GaussianProcess(zm, mc)
cgp = ConditionedGaussianProcess(gp, ip)
bop = BayesOpt(cgp, acq = "UCB")

bop.augment()
bop.augment()

print("fine")
