# NAME: 'ptSets.py'
#
# PURPOSE: Collection of different strategies to construct pointsets
#
# DESCRIPTION: see PURPOSE; Load generating vector via
#
# AUTHOR: NK, kraemer(at)ins.uni-bonn.de

import numpy as np



# The pointset is based on a randomly shifted lattice rule with product weight $\gamma_j = 1/j^2$,
# based on the generating vector from Frances Kuo's website
# http://web.maths.unsw.edu.au/(SIMSYMBOL)fkuo/ as ``lattice-39102-1024-1048576.3600''.
def getPtsLattice(numPts, dim, randShift = True):

	genVec = np.loadtxt('.genVecs/vec.txt')
	genVec = genVec[0:dim, 1]
	ptSet = np.zeros((numPts, dim))

	if randShift == True:
		shift = np.random.rand(dim)
		for idx in range(numPts):
			ptSet[idx,:] = (1.0*genVec * idx / numPts  + shift)% 1.0
	else:
		for idx in range(numPts):
			ptSet[idx,:] = (1.0*genVec * idx / numPts)% 1.0

	return ptSet

# stolen from https://laszukdawid.com/2017/02/04/halton-sequence-in-python/
def getPtsHalton(size, dim):
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
		seq.append([vanDerCorput(i, base) for i in range(size)])
	return np.array(seq).T

















