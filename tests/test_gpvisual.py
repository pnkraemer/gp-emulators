
from gpemu.gpvisual import *
from gpemu.gaussianprocesses import *
from gpemu.data import *


np.random.seed(1)

gp = StandardGP()
gp_data = ToyGPData1d(num_pts = 3)
cgp = ConditionedGaussianProcess(gp, gp_data)

gpvis = GPVisual(cgp)
gpvis.addplot_mean()
gpvis.addplot_truth()
gpvis.addplot_deviation()
gpvis.addplot_samples()
gpvis.addplot_observations()
plt.legend()
plt.show()

gpvis2 = GPVisual(cgp)
gpvis2.addplot_mean()
gpvis2.addplot_fancy_deviation(num_shades = 50)
gpvis2.addplot_truth()
gpvis2.addanimation_samples()
gpvis2.addplot_observations()
plt.legend()
plt.show()


gpvis3 = NakedGPVisual(cgp)
gpvis3.addplot_mean()
gpvis3.addplot_deviation()
gpvis3.addplot_samples()
gpvis3.addanimation_samples()
gpvis3.addplot_observations()
plt.show()

print("\nAll seems fine\n")
