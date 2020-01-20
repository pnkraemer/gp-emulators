import sys
sys.path.insert(0, "../modules")

from pointsets import *
from covariances import *
from gpvisual import *
from gaussianprocesses import *
from data import *
import numpy as np
import scipy.linalg as spl

np.random.seed(1)

num_pts = 225
dim = 1
ptset1 = Halton.construct(num_pts, dim)
num_pts2 = 400
ptset2 = Halton.construct(num_pts2, dim)

cov_mtrx = MaternCov.fast_mtrx(ptset1, ptset2, 10.0)

x = 10 * np.random.rand(2)
N = 10
mtrx = np.vander(x, N)

x = np.random.rand(225)
y = np.random.rand(400)
mtrx = spl.toeplitz(x,y)

fig = plt.figure()
plt.axis('off')
plt.imshow(mtrx, cmap='copper', interpolation='nearest')
plt.show()

fig.savefig('titleimg.jpg', format='jpg', dpi=480, bbox_inches = "tight")

#gp = StandardGP()
#gp_data = ToyGPData1d(num_pts = 4)
#cgp = ConditionedGaussianProcess(gp, gp_data)



#gpvis = NakedGPVisual(cgp, ctheme = "darkgray", bgroundtheme = "lightgray")
#gpvis.addplot_mean()
#gpvis.addplot_deviation()
#gpvis.addplot_observations()
#plt.show()

