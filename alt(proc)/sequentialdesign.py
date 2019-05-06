"""
NAME: sequentialdesign.py

PURPOSE: first attempt to check sinsbeck/nowak approach
"""
import numpy as np
import sys
sys.path.insert(0, "./modules")
from data import ToyInverseProblem
import sympy

np.random.seed(2)

def functional(GaussProc, InvProb, MonCar):

    def integrand(GaussProc, InvProb):
        return 0

    return 0



def EV(eta, alpha, sigma, c, a):
    left_nom = np.exp(-((2. * a**2 * c)/(4. * alpha**2 * c * eta**2 + 4. * c * sigma**2 + 1.)))
    left_denom = np.sqrt(4. * alpha**2 * c * eta**2 + 4. * c * sigma**2 + 1.)
    right_nom = np.exp(-((2. * a**2 * c)/(4. * alpha**2 * c * eta**2 + 2. * c * sigma**2 + 1.)))
    right_denom = np.sqrt(4. * alpha**2 * c * eta**2 + 2. * c * sigma**2 + 1.)
    return left_nom/left_denom - right_nom/right_denom

from covariances import ExpCov, MaternCov, GaussCov
from data import ToyInverseProblem
from montecarlo import MonteCarlo
from pointsets import Random, Mesh1d




cov_fct = MaternCov()
IP = ToyInverseProblem(0.1)
dim = 1
num_pts_mc = 10000
monte_carlo = MonteCarlo(num_pts_mc, dim)

#pt_x = Random(1,1)
# pt_z = Random(1,1)
# print(pt_x.points, pt_z.points)

print(IP.true_observations)
print(IP.locations.points)

def integration(pt_x, cov_fct, IP, monte_carlo):

    def integrand(pt_z, pt_x = pt_x, cov_fct = cov_fct, IP = IP):
        eta = cov_fct.assemble_entry_cov_mtrx(pt_x, pt_x)
        alpha = cov_fct.assemble_entry_cov_mtrx(pt_z, pt_x) / eta
        sigma = cov_fct.assemble_entry_cov_mtrx(pt_z, pt_z) - alpha * cov_fct.assemble_entry_cov_mtrx(pt_x, pt_z)
        c = 1./(2.*IP.variance)
        a = IP.observations
        return EV(eta, alpha, sigma, c, a)
    
    return monte_carlo.compute_integral(integrand)

x = Mesh1d(10)
results = np.zeros((10,1))
for i in range(10):
    results[i, 0] = integration(x.points[i,:], cov_fct, IP, monte_carlo)
    print(results)





def sine(pt):
	return np.sin(5*pt)
def likelihood(pt, IP):
	A = sine(pt)
	y = IP.observations
	v = IP.variance
	B = np.exp(-(y - sine(pt))**2/(2*v))
	return B

x1 = np.linspace(0, 1, 500)
x2 = likelihood(x1, IP)


import matplotlib.pyplot as plt
plt.style.use("ggplot")
plt.grid(True)

plt.plot(x.points, 1000 * results, 'o')
plt.plot(x1, x2[0,:], '-')

plt.show()


















# # Expected value of e^(-c(a-gamma*beta)^2), beta is N(alpha mu, sigma^2) distributed
# def expected_value(c = sympy.Symbol('c'), 
#                    a = sympy.Symbol('a'),
#                    gamma = sympy.Symbol('gamma'),
#                    mu = sympy.Symbol('mu'),
#                    sigma  = sympy.Symbol('sigma'),
#                    alpha = sympy.Symbol('alpha')):
#     numerator = sympy.exp(-c * (a - gamma * alpha * mu)**2/(2.*c*gamma**2*sigma**2 + 1.))
#     denominator = sympy.sqrt(2. * gamma**2 * sigma**2 * c + 1.)
#     return sympy.simplify(numerator/denominator)

# # Variance of e^(-c(a-gamma*beta)^2), beta is N(alpha mu, sigma^2) distributed
# # Using formula V(f(x)) = E(f^2(x)) - E(f(x))^2
# def variance(c = sympy.Symbol('c'), 
#              a = sympy.Symbol('a'),
#              gamma = sympy.Symbol('gamma'),
#              mu = sympy.Symbol('mu'),
#              sigma  = sympy.Symbol('sigma'),
#              alpha = sympy.Symbol('alpha')):
#     left_summand = expected_value(c = 2*sympy.Symbol('c'), 
#                                   a = sympy.Symbol('a'),
#                                   gamma = 1.,
#                                   mu = sympy.Symbol('mu'),
#                                   sigma  = sympy.Symbol('sigma'),
#                                   alpha = sympy.Symbol('alpha'))
#     right_summand = (expected_value(c = sympy.Symbol('c'), 
#                                     a = sympy.Symbol('a'),
#                                     gamma = 1.,
#                                     mu = sympy.Symbol('mu'),
#                                     sigma  = sympy.Symbol('sigma'),
#                                     alpha = sympy.Symbol('alpha')))**2
#     return sympy.simplify(left_summand - right_summand)



# var = variance()

# c = 2*sympy.Symbol('c') / (2. * sympy.Symbol('c') * sympy.Symbol('sigma')**2 + 1.)
# a = sympy.Symbol('a')
# gamma = sympy.Symbol('alpha')
# mu = 0
# sigma = sympy.Symbol('eta')
# alpha = 0
# numerator = expected_value(c, a, gamma, mu, sigma, alpha)
# denominator = 2. * sympy.Symbol('c') * sympy.Symbol('sigma')**2 + 1.
# leftsummand = sympy.simplify(numerator/denominator)
# print("left_summand =", leftsummand)




































# # sequential design method
# def integrand(a, c, alpha, eta, sigma):
# 	A = 2*c*a**2 / (alpha**2 * eta**2 * 1./c + 4 * c * sigma**2 + 1)
# 	A = np.exp(-A)
# 	B = 4 * alpha**2 * eta**2 * c + 4 * sigma**2 * c + 1
# 	B = np.sqrt(B)
	
# 	C = 2 * c * a**2 / (4 * c * alpha**2 * eta**2 + 2 * c * sigma**2 + 1)
# 	C = np.exp(-C)
# 	D = 4 * alpha**2 * eta**2 * c * (2 * c * sigma**2 + 1) + (2 * sigma**2 * c + 1)**2
# 	D = np.sqrt(D)
# 	return A/B - C/D


# IP = ToyInverseProblem()









