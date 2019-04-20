import sys
sys.path.insert(0, "../modules")
from means import *
from pointsets import *

num_pts = 10
dim = 1
ptset = Random.construct(num_pts, dim)

zero_mean = ZeroMean()
meanvec_zero = zero_mean.evaluate(ptset)
assert(meanvec_zero.shape == (num_pts, 1))
assert(meanvec_zero[0,0] == 0.)

const = 1.425
const_mean = ConstMean(const)
meanvec_const = const_mean.evaluate(ptset)
assert(meanvec_const.shape == (num_pts, 1))
assert(meanvec_const[0,0] == const)

print("\nCase (%i, %i, %.1e) successful" %(num_pts, dim, const))

num_pts = 2
dim = 9
ptset = Random.construct(num_pts, dim)

zero_mean = ZeroMean()
meanvec_zero = zero_mean.evaluate(ptset)
assert(meanvec_zero.shape == (num_pts, 1))
assert(meanvec_zero[0,0] == 0.)

const = 1.42525342597
const_mean = ConstMean(const)
meanvec_const = const_mean.evaluate(ptset)
assert(meanvec_const.shape == (num_pts, 1))
assert(meanvec_const[0,0] == const)

print("\nCase (%i, %i, %.1e) successful\n" %(num_pts, dim, const))

print("\nAll seems good\n\n")

