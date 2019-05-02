import sys
sys.path.insert(0, "../modules")
from pointsets import *

num_pts = 10000
dim = 1

print("\nChecking pointsets:")
"""
Check 1d
"""
print("\td=1...", end="")
mesh1d = Mesh1d.construct(num_pts)
random = Random.construct(num_pts, dim)
lattice = Lattice.construct(num_pts, dim)
assert(mesh1d.shape==random.shape)
assert(mesh1d.shape==lattice.shape)
assert(random.shape==lattice.shape)
assert(random.shape==(num_pts, dim))
print("successful!")
halton = Halton.construct(num_pts, dim)
assert(halton.shape==(num_pts, dim))

"""
Check 5d
"""
print("\td=5...", end="")
dim = 5
random = Random.construct(num_pts, dim)
lattice = Lattice.construct(num_pts, dim)
assert(random.shape==lattice.shape)
assert(random.shape==(num_pts, dim))
print("successful!")

print("\nCheck Halton...", end = "")
halton = Halton.construct_withzero(200, 2)
halton2 = Halton.construct(200, 2)
assert(len(halton) == len(halton2))
print("successful!")


