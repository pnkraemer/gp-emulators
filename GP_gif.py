
import numpy as np

sys.path.insert(0, "./modules")
from pointsets import PointSet, Random, Mesh1d
from covariances import GaussCov, ExpCov, MaternCov
from means import ZeroMean
from data import ToyGPData
from gaussianprocesses import GaussianProcess, ConditionedGaussianProcess

from matplotlib import pyplot as plt
from matplotlib import animation

np.random.seed(1)


def animation_gp(gaussproc, string, num_pts = 20):
	# set up plotting parameters
	mesh = Mesh1d(num_pts)
	meanvec = gaussproc.mean_fct.assemble_mean_vec(mesh.points)
	covmat = gaussproc.cov_fct.assemble_cov_mtrx(mesh.points, mesh.points)
#	print("meshpoints =", mesh.points)
#	print("meanvec =", meanvec)
#	print("covmat =", covmat)
	meanvec_upper = meanvec.T + 2*np.sqrt(np.abs(np.diag(covmat)))
	meanvec_lower = meanvec.T - 2*np.sqrt(np.abs(np.diag(covmat)))

	# Plot initial configuration
	plt.style.use("ggplot")
	plt.rcParams["figure.figsize"] = [8,4]
	fig = plt.figure()
	ax = plt.axes(xlim = [0,1], ylim = [-2.5,2.5])
	line, = ax.plot([], [])
	ax.fill_between(mesh.points[:,0], meanvec_lower[0,:], meanvec_upper[0,:], alpha = 0.3)
	ax.plot(mesh.points, meanvec, color = "black", linewidth = 2)
	
	ax.plot(data.locations.points, data.observations, 'o', color = "black")

	# Prepare animation
	def init():
	    line.set_data([], [])
	    return line,

	# animation function.  This is called sequentially
	def animate(i):
	    x = mesh.points
	    samples = gaussproc.sample(mesh)
	    line.set_data(x, samples)
	    line.set_linewidth(3)
	    return line,

	# Call animation
	anim = animation.FuncAnimation(fig, animate, init_func=init,
	                               frames=25, interval=500, blit=True)
	#plt.show()
	anim.save(string, fps = 5, dpi = 200, writer="ffmpeg")#, fps=30, extra_args=['-vcodec', 'libx264'])














# Set up problem and Gaussian process approximation
num_evals = 3
noise = 0.
data = ToyGPData(num_evals, noise)
zero_mean = ZeroMean()
cov_fct = MaternCov(1.0)
gp = GaussianProcess(zero_mean, cov_fct)
cGP = ConditionedGaussianProcess(gp, data)



num_pts = 200
animation_gp(gp, '../animations/animation_prior.mp4', num_pts)
animation_gp(cGP, '../animations/animation_posterior.mp4', num_pts)




