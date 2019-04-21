"""
NAME: gpvisual.py

"""

from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np

from pointsets import Mesh1d
from gaussianprocesses import *

class GPVisual():

    def __init__(self, GaussProc, ctheme = "darkslategray", num_pts = 200, 
                 xlim = [0,1], title = "", naked = False):
        plt.style.use("ggplot")
        plt.rcParams["figure.figsize"] = [10,5]
        plt.rcParams["lines.linewidth"] = 1
        plt.rcParams["lines.markersize"] = 8
        plt.rcParams["grid.linewidth"] = 1
        plt.rcParams["font.size"] = 16
        plt.rcParams['xtick.major.width']= 0
        plt.rcParams['ytick.major.width']= 0
        if naked == True:
            plt.rcParams["axes.facecolor"] = "white"
        figure = plt.figure()
        axis = plt.axes()
        plt.title(title, color = "black", alpha = 0.6)
        plt.xlim(xlim)
        plt.grid(True)
        self.fig = figure
        self.ax = axis
        self.gp = GaussProc
        self.num_pts = num_pts
        self.mesh = Mesh1d.construct(self.num_pts)
        self.mean_vec = self.gp.mean_fct.evaluate(self.mesh)
        self.color = ctheme

    def addplot_mean(self):
        self.ax.plot(self.mesh, self.mean_vec, color = self.color, 
                     label = "Mean function")


    def addplot_truth(self):
        assert(self.gp.is_conditioned==True), "Not a conditioned GP!"
        self.ax.plot(self.mesh, self.gp.data.forward_map(self.mesh) , color = "darkred", linestyle = "dashed", linewidth = 2, 
                     label = "True function")


    def addplot_deviation(self, num_dev = 2):
        cov_mtrx = self.gp.cov_fct.evaluate(self.mesh, self.mesh)
        pos_dev = self.mean_vec.T + 2*num_dev*np.sqrt(np.abs(np.diag(cov_mtrx)))
        neg_dev = self.mean_vec.T - 2*num_dev*np.sqrt(np.abs(np.diag(cov_mtrx)))
        self.ax.fill_between(self.mesh[:,0], neg_dev[0,:], pos_dev[0,:], 
                             facecolor = self.color, linewidth = 1, linestyle = "-", 
                             alpha = 0.3, label = "Confidence interval")

    def addplot_fancy_deviation(self, num_dev = 3):
        cov_mtrx = self.gp.cov_fct.evaluate(self.mesh, self.mesh)
        num_shades = 50
        shade = 1.0/num_shades
        for i in range(num_shades):
            pos_dev = self.mean_vec.T + i*shade*num_dev*np.sqrt(np.abs(np.diag(cov_mtrx)))
            neg_dev = self.mean_vec.T - i*shade*num_dev*np.sqrt(np.abs(np.diag(cov_mtrx)))
            self.ax.fill_between(self.mesh[:,0], neg_dev[0,:], pos_dev[0,:], 
                                 facecolor = self.color, linewidth = 1, linestyle = "-", 
                                 alpha = shade)

    def addplot_samples(self, num_samp = 5):
        for i in range(num_samp):
            samp = self.gp.sample(self.mesh)
            self.ax.plot(self.mesh, samp, '-', color = 0.5*np.random.rand(3,))

    def addplot_observations(self):
        assert(self.gp.is_conditioned==True), "Not a conditioned GP!"
        locations = self.gp.data.locations
        observations = self.gp.data.observations
        self.ax.plot(locations, observations, 'o', color = "white")
        if self.gp.data.variance == 0:
            self.ax.plot(locations, observations, 'o', color = self.color,
                         markerfacecolor = "white", markeredgecolor = self.color, 
                         markeredgewidth = 1, label = "Observations")
        else:
            self.ax.errorbar(locations, observations, 
                         yerr = np.sqrt(self.gp.data.variance), color = self.color, 
                         fmt='o', markerfacecolor = "white", 
                         markeredgecolor = self.color, markeredgewidth = 1, 
                         capsize = 3, label = "Observations")


    def addanimation_samples(self):

        def init():
            line.set_data([], [])
            return line,

        def animate(i):
            samp = self.gp.sample(self.mesh)
            line.set_data(self.mesh, samp)
            line.set_linewidth(1)
            line.set_color(0.4*np.random.rand(3,))
            return line,

        line, = self.ax.plot([], [])
        self.anim = animation.FuncAnimation(self.fig, animate, init_func=init,
                                       frames=25, interval=500, blit=True)



class NakedGPVisual(GPVisual):
    def __init__(self, GaussProc, ctheme = "darkslategray", num_pts = 200, xlim = [0,1], title = ""):
        GPVisual.__init__(self, GaussProc, ctheme = ctheme, num_pts = 200, xlim = [0,1], 
                          title = "", naked = True)
        plt.grid(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.tick_params(top=False, bottom=False, left=False, right=False, 
                        labelleft=False, labelbottom=False)

