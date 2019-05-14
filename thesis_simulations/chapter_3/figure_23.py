import sys
sys.path.insert(0, "../../modules")

from covariances import *
from pointsets import *

np.random.seed(1)

num_plt_pts = 200

plt_pts = 5*Mesh1d.construct(num_plt_pts) - 2.5

matern025 = MaternCov(smoothness = 1.5, corr_length = 0.25)
matern2 = MaternCov(smoothness = 1.5, corr_length = 2.0)

values_025 = matern025.evaluate(plt_pts, 0.*np.ones((1,1)))
values_2 = matern2.evaluate(plt_pts, 0.*np.ones((1,1)))

import matplotlib.pyplot as plt 
plt.style.use("fivethirtyeight")
plt.plot(plt_pts, values_025, '-')
plt.plot(plt_pts, values_2, '-')
plt.show()

print("\nMatern 0.25:")
for i in range(num_plt_pts):
	print("(", plt_pts[i,0], ",", values_025[i,0], ")")

print("\nMatern 2:")
for i in range(num_plt_pts):
	print("(", plt_pts[i,0], ",", values_2[i,0], ")")

