from __future__ import division	# division of integers into decimal numbers

import numpy as np
import matplotlib.pyplot as plt

from fem_new import forward_operator_fem_1d


# determine FEM parameters
h = 1.0/1234
a = np.array([1.0, 1.0, 2.0])
J = np.linspace(0,1,111)

# solve fem problem and evaluate at the J-points
UJ = forward_operator_fem_1d(a, h, J)


# plot solution
plt.plot(J, UJ, '-', linewidth = 3, color = "darkslategray")
plt.grid()
plt.title("FEM Solution in 1D")
plt.show()

