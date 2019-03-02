# FILENAME: ptSetPlayGround.py
#
# PURPOSE: play around with a pointset class 
#
# DESCRIPTION: create a pointset class and some associated functions
#
# LAST RUN WITH: Python 3.7.2 
#
# AUTHOR: NK

import numpy as np


class ptSet:

	def __init__(self, entries):
		self.entries = entries
		self.numPts = len(entries)
		self.dim = len(entries.T)
		self.isSubSet = False
		
	def createSubset(self, idxFrom, idxTo):
		subSet = ptSet(self.entries[idxFrom:idxTo])
		subSet.isSubSet = True
		return subSet

	def makeRandom(numPts, dim):
		randSet = np.random.rand(numPts, dim)
		return ptSet(randSet)

	# based on the generating vector from Frances Kuo's website
	# http://web.maths.unsw.edu.au/(SIMSYMBOL)fkuo/ as 
	# ``lattice-39102-1024-1048576.3600''.
	def makeLattice(numPts, dim, randShift = False):

		genVec = np.loadtxt('genVecs/vec.txt')
		genVec = genVec[0:dim, 1]
		latticeSet = np.zeros((numPts, dim))

		if randShift == True:
			shift = np.random.rand(dim)
			for idx in range(numPts):
				latticeSet[idx,:] = (1.0*genVec * idx / numPts  + shift)% 1.0
		else:
			for idx in range(numPts):
				latticeSet[idx,:] = (1.0*genVec * idx / numPts)% 1.0
		
		return ptSet(latticeSet)

	# does not include the first halton point (0,0)
	# stolen from https://laszukdawid.com/2017/02/04/halton-sequence-in-python/
	def makeHalton(numPts, dim):

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
			seq.append([vanDerCorput(i, base) for i in range(numPts)])
		return ptSet(np.array(seq).T)



haltonSet = ptSet.makeHalton(11,2)
print(haltonSet.entries)











print("\n\n----------------------------------")
print("Testing...")
print("----------------------------------\n\n")
a = np.random.rand(4)
haltonPts = ptSet(a)
print(haltonPts.entries)
print(haltonPts.numPts)
print(haltonPts.dim)
print(haltonPts.isSubSet)
print()

haltonSubPts = haltonPts.createSubset(1,2)
print(haltonSubPts.entries)
print(haltonSubPts.numPts)
print(haltonSubPts.dim)
print(haltonSubPts.isSubSet)
print()

haltonSubPts.entries[0] = 2
print(haltonPts.entries)
print(haltonSubPts.entries)
print()


randSet = ptSet.makeRandom(10,2)
print(randSet.entries)
print()

latticePtSet = ptSet.makeLattice(10, 2, False)
print(latticePtSet.entries)
print()






