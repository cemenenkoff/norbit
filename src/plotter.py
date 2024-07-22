from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import set_matplotlib_formats
from mpl_toolkits.mplot3d import Axes3D
from orbiter import Orbiter

set_matplotlib_formats("pdf", "png")
plt.style.use("classic")  # Use a serif font.
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["figure.autolayout"] = False
plt.rcParams["figure.figsize"] = 10, 6
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["font.size"] = 10
plt.rcParams["lines.linewidth"] = 2.0
plt.rcParams["lines.markersize"] = 6
plt.rcParams["legend.fontsize"] = 14
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
matplotlib.rcParams["xtick.direction"] = "out"
matplotlib.rcParams["ytick.direction"] = "out"


class Plotter:
    def __init__(self, solved_orbiter: Orbiter):
        self.N = solved_orbiter.N
        self.t = solved_orbiter.t
        self.r = solved_orbiter.r
        self.colors = solved_orbiter.colors
        self.mf = 1
        self.alpha = 0.7  # Global transparency value to see where orbits overlap.
        # Set low and high bounds for arrays used in the phase space figures to ensure
        # the system is in a "settled down" dynamic equilibrium.
        self.lo = int(0.35 * (len(self.t)))
        self.hi = int(0.95 * (len(self.t)))
        self.labels = [r"$m_" + str(i) + "$" for i in range(self.N)]
        self.outfolder = solved_orbiter.outfolder
        self.outfolder.mkdir(parents=True, exist_ok=True)

    def plot_3d_trajectories(self):
        fig = plt.figure(1, facecolor="white")  # 3D plot of orbital trajectories.
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        plt.title(
            r"$\mathrm{%s}\ $" % str(self.N) + r"$\mathrm{Orbiting\ Bodies}$", y=1.05
        )
        ax.set_xlabel(r"$\mathrm{x}\ \mathrm{(m)}$", labelpad=10)
        ax.set_ylabel(r"$\mathrm{y}\ \mathrm{(m)}$", labelpad=10)
        ax.set_zlabel(r"$\mathrm{z}\ \mathrm{(m)}$", labelpad=10)
        # Adjust the offset text position
        # ax.get_xaxis().get_offset_text().set_position((0, 1.05))
        # ax.get_yaxis().get_offset_text().set_position((0, 1.05))
        # ax.get_zaxis().get_offset_text().set_position((0, 1.05))
        for i in range(self.N):  # For all times, plot mi's (x,y,z) data.
            ax.plot(
                self.r[:, i, 0],
                self.r[:, i, 1],
                self.r[:, i, 2],
                color=self.colors[list(self.colors.keys())[i]],
                label=self.labels[i],
                alpha=self.alpha,
            )
        ax.axis("equal")
        plt.legend(loc="upper left")
        outfile = "3d_trajectories"
        outpath = self.outfolder / outfile
        plt.savefig(outpath)
