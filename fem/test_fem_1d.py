from __future__ import division	# division of integers into decimal numbers

import numpy as np
import matplotlib.pyplot as plt

from fem_1d import forward_operator_fem_1d


# determine FEM parameters
print "\nWhich mesh-width h? (e.g. h = 1.0/1024)"
h = input("Enter:  ")
print ""

# Determine parameters
a = np.array([1.0, 1.0, 2.0])
J = np.linspace(0,1,111)

# solve fem problem and evaluate at the J-points
UJ = forward_operator_fem_1d(a, h, J)


# plot solution
plt.plot(J, UJ, '-', linewidth = 3, color = "darkslategray")
plt.grid()
plt.title("FEM Solution in 1D")
plt.show()

