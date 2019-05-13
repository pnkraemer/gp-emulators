import sys
sys.path.insert(0, "../../modules")

from covariances import *
from pointsets import *

np.random.seed(1)

num_plt_pts = 200

plt_pts = 5*Mesh1d.construct(num_plt_pts) - 2.5

matern075 = MaternCov(0.75)
matern4 = MaternCov(4)

values_075 = matern075.evaluate(plt_pts, 0.*np.ones((1,1)))
values_4 = matern4.evaluate(plt_pts, 0.*np.ones((1,1)))

import matplotlib.pyplot as plt 
plt.style.use("fivethirtyeight")
plt.plot(plt_pts, values_075, '-')
plt.plot(plt_pts, values_4, '-')
plt.show()

print("\nMatern 0.75:")
for i in range(num_plt_pts):
	print("(", plt_pts[i,0], ",", values_075[i,0], ")")

print("\nMatern 4:")
for i in range(num_plt_pts):
	print("(", plt_pts[i,0], ",", values_4[i,0], ")")

