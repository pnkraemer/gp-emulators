"""
NAME: sequentialdesign.py

PURPOSE: first attempt to check sinsbeck/nowak approach
"""
import numpy as np
import sys
sys.path.insert(0, "./modules")
from pointsets import Random
from data import ToyInverseProblem
import sympy

# Expected value of e^(-c(a-gamma*beta)^2), beta is N(alpha mu, sigma^2) distributed
def expected_value(c = sympy.Symbol('c'), 
                   a = sympy.Symbol('a'),
                   gamma = sympy.Symbol('gamma'),
                   mu = sympy.Symbol('mu'),
                   sigma  = sympy.Symbol('sigma'),
                   alpha = sympy.Symbol('alpha')):
    numerator = sympy.exp(-c * (a - gamma * alpha * mu)**2/(2.*c*gamma**2*sigma**2 + 1.))
    denominator = sympy.sqrt(2. * gamma**2 * sigma**2 * c + 1.)
    return sympy.simplify(numerator/denominator)

# Variance of e^(-c(a-gamma*beta)^2), beta is N(alpha mu, sigma^2) distributed
# Using formula V(f(x)) = E(f^2(x)) - E(f(x))^2
def variance(c = sympy.Symbol('c'), 
             a = sympy.Symbol('a'),
             gamma = sympy.Symbol('gamma'),
             mu = sympy.Symbol('mu'),
             sigma  = sympy.Symbol('sigma'),
             alpha = sympy.Symbol('alpha')):
    left_summand = expected_value(c = 2*sympy.Symbol('c'), 
                                  a = sympy.Symbol('a'),
                                  gamma = 1.,
                                  mu = sympy.Symbol('mu'),
                                  sigma  = sympy.Symbol('sigma'),
                                  alpha = sympy.Symbol('alpha'))
    right_summand = (expected_value(c = sympy.Symbol('c'), 
                                    a = sympy.Symbol('a'),
                                    gamma = 1.,
                                    mu = sympy.Symbol('mu'),
                                    sigma  = sympy.Symbol('sigma'),
                                    alpha = sympy.Symbol('alpha')))**2
    return sympy.simplify(left_summand - right_summand)



var = variance()

c = 2*sympy.Symbol('c') / (2. * sympy.Symbol('c') * sympy.Symbol('sigma')**2 + 1.)
a = sympy.Symbol('a')
gamma = sympy.Symbol('alpha')
mu = 0
sigma = sympy.Symbol('eta')
alpha = 0
numerator = expected_value(c, a, gamma, mu, sigma, alpha)
denominator = 2. * sympy.Symbol('c') * sympy.Symbol('sigma')**2 + 1.
leftsummand = sympy.simplify(numerator/denominator)
print("left_summand =", leftsummand)




































# sequential design method
def integrand(a, c, alpha, eta, sigma):
	A = 2*c*a**2 / (alpha**2 * eta**2 * 1./c + 4 * c * sigma**2 + 1)
	A = np.exp(-A)
	B = 4 * alpha**2 * eta**2 * c + 4 * sigma**2 * c + 1
	B = np.sqrt(B)
	
	C = 2 * c * a**2 / (4 * c * alpha**2 * eta**2 + 2 * c * sigma**2 + 1)
	C = np.exp(-C)
	D = 4 * alpha**2 * eta**2 * c * (2 * c * sigma**2 + 1) + (2 * sigma**2 * c + 1)**2
	D = np.sqrt(D)
	return A/B - C/D


IP = ToyInverseProblem()









