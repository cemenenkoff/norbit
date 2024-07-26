from typing import Literal

import matplotlib
import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import set_matplotlib_formats
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
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
    """Create plots with a solved (i.e. already-run) `Orbiter` object's data.

    Attributes:
        N
        t
        r
        p
        ke
        ke_tot
        ke_sys
        pe
        pe_tot
        pe_sys
        colors
        mf
        mf_label
        alpha
        lo
        hi
        labels
        outfolder

    """

    def __init__(self, solved_orbiter: Orbiter) -> None:
        """Instantiate a `Plotter` to plot the data contained in the `solved_orbiter`.

        Args:
            solved_orbiter (Orbiter): An `Orbiter` object that has been fully solved.
        """
        self.N = solved_orbiter.N
        self.t = solved_orbiter.t
        self.r = solved_orbiter.r
        self.p = solved_orbiter.p
        self.ke = solved_orbiter.ke
        self.ke_tot = solved_orbiter.ke_tot
        self.ke_sys = solved_orbiter.ke_sys
        self.pe = solved_orbiter.pe
        self.pe_tot = solved_orbiter.pe_tot
        self.pe_sys = solved_orbiter.pe_sys
        self.colors = solved_orbiter.colors
        self.mf = 1
        self.mf_label = r"$m_{%s}\ $" % str(1)
        self.alpha = 0.7  # Global transparency value to see where orbits overlap.
        # Set low and high bounds for arrays used in the phase space figures to ensure
        # the system is in a "settled down" dynamic equilibrium.
        self.lo = int(0.35 * (len(self.t)))
        self.hi = int(0.95 * (len(self.t)))
        self.labels = [r"$m_" + str(i) + "$" for i in range(self.N)]
        self.outfolder = solved_orbiter.outfolder
        self.outfolder.mkdir(parents=True, exist_ok=True)

    def animate_3d_orbits(self):
        df = pd.DataFrame(
            {
                "time": self.t,
                "x": self.r[: len(self.t), 0, 0],
                "y": self.r[: len(self.t), 0, 1],
                "z": self.r[: len(self.t), 0, 2],
            }
        )
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        title = ax.set_title("3D Test")

        # Initialize scatter plot
        graph = ax.scatter([], [], [], c="blue", marker="o", s=1)

        def update_graph(num):
            # Filter the data for the current frame
            data = df.iloc[num]
            # Update the scatter plot
            graph._offsets3d = ([data.x], [data.y], [data.z])
            # Update the title
            title.set_text(f"3D Test, time={num}")
            return graph, title

        # Create animation
        ani = FuncAnimation(
            fig, update_graph, frames=len(self.t), interval=40, blit=True
        )

        # Save the animation
        outfile = "3d_orbits.gif"
        outpath = self.outfolder / outfile
        writer = PillowWriter(fps=20)
        plt.legend(loc="center left")
        plt.close(fig)
        ani.save(outpath, writer=writer)


    # def animate_3d_orbits(self) -> None:
    #     fig = plt.figure(facecolor="white")
    #     ax = fig.add_subplot(1, 1, 1, projection="3d")
    #     plt.title(
    #         r"$\mathrm{%s}\ $" % str(self.N)
    #         + r"$\mathrm{Orbiting\ Bodies,\ }$"
    #         + r"$\mathrm{Position\ vs.\ Time}$",
    #         y=1.05,
    #     )
    #     ax.set_xlabel(r"$\mathrm{x-Position\ (m)}$", labelpad=10)
    #     ax.set_ylabel(r"$\mathrm{y-Position\ (m)\quad\text{\ \ \ }}$", labelpad=10)
    #     ax.set_zlabel(r"$\mathrm{z-Position\ (m)\quad\text{\ \ \ }}$", labelpad=10)

    #     lines = [
    #         ax.plot(
    #             [],
    #             [],
    #             [],
    #             color=self.colors[list(self.colors.keys())[i]],
    #             label=self.labels[i],
    #             alpha=self.alpha,
    #         )[0]
    #         for i in range(self.N)
    #     ]

    #     def init():
    #         for line in lines:
    #             line.set_data([], [])
    #             line.set_3d_properties([])
    #         return lines

    #     def update(frame):
    #         for i, line in enumerate(lines):
    #             line.set_data(self.r[:frame, i, 0], self.r[:frame, i, 1])
    #             line.set_3d_properties(self.r[:frame, i, 2])
    #         return lines

    #     anim = FuncAnimation(fig, update, frames=len(self.r), init_func=init, blit=True)

    #     outfile = "3d_orbits.gif"
    #     outpath = self.outfolder / outfile
    #     writer = PillowWriter(fps=20)
    #     anim.save(outpath, writer=writer)
    #     plt.legend(loc="center left")
    #     plt.show()

    def plot_3d_orbits(self) -> None:
        """Plot the 3D orbital paths of N mutually-interacting gravitational bodies."""
        fig = plt.figure(facecolor="white")
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        plt.title(
            r"$\mathrm{%s}\ $" % str(self.N)
            + r"$\mathrm{Orbiting\ Bodies,\ }$"
            + r"$\mathrm{Position\ vs.\ Time}$",
            y=1.05,
        )
        ax.set_xlabel(r"$\mathrm{x-Position\ (m)}$", labelpad=10)
        ax.set_ylabel(r"$\mathrm{y-Position\ (m)\quad\text{\ \ \ }}$", labelpad=10)
        ax.set_zlabel(r"$\mathrm{z-Position\ (m)\quad\text{\ \ \ }}$", labelpad=10)
        for i in range(self.N):
            ax.plot(
                self.r[:, i, 0],
                self.r[:, i, 1],
                self.r[:, i, 2],
                color=self.colors[list(self.colors.keys())[i]],
                label=self.labels[i],
                alpha=self.alpha,
            )
        ax.axis("equal")
        plt.legend(loc="center left")
        outfile = "3d_orbits"
        outpath = self.outfolder / outfile
        plt.savefig(outpath)

    def plot_3d_orbits_viewed_from_pos_axis(self, axis: Literal["x", "y", "z"]) -> None:
        """Plot a 2D slice of the 3D orbital paths of the N bodies.

        Args:
            axis (Literal["x", "y", "z"]): The axis to look down along (from the
                positive direction).
        """
        fig = plt.figure(facecolor="white")
        ax = fig.add_subplot(111)
        plt.title(
            "$\mathrm{%s}\ $" % str(self.N)
            + r"$\mathrm{Orbiting\ Bodies,\ }$"
            + r"$\mathrm{Position\ vs.\ Time\ }$"
            + r"$\mathrm{Viewed \ from\ the\ \mathrm{+}\ }$"
            + "$\mathrm{%s}\ $" % axis
            + r"$\mathrm{-Axis}$",
            y=1.05,
        )
        if axis == "x":
            ax.set_xlabel(r"$\mathrm{y-Position\ (m)}$")
            ax.set_ylabel(r"$\mathrm{z-Position\ (m)}$")
            ind1, ind2 = 1, 2
        if axis == "y":
            ax.set_xlabel(r"$\mathrm{x-Position\ (m)}$")
            ax.set_ylabel(r"$\mathrm{z-Position\ (m)}$")
            ind1, ind2 = 0, 2
        if axis == "z":
            ax.set_xlabel(r"$\mathrm{x-Position\ (m)}$")
            ax.set_ylabel(r"$\mathrm{y-Position\ (m)}$")
            ind1, ind2 = 0, 1
        for i in range(self.N):
            ax.plot(
                self.r[:, i, ind1],
                self.r[:, i, ind2],
                color=self.colors[list(self.colors.keys())[i]],
                label=self.labels[i],
                alpha=self.alpha,
            )
        ax.axis("equal")
        ax.legend(loc="lower right")
        outfile = f"3d_orbits_viewed_from_pos_{axis}_axis"
        outpath = self.outfolder / outfile
        plt.savefig(outpath)

    def plot_ke_tot_vs_t(self) -> None:
        """Plot the total kinetic energy over time for each of the N bodies."""
        fig = plt.figure(facecolor="white")
        ax = fig.add_subplot(111)
        plt.title(
            "$\mathrm{%s}\ $" % str(self.N)
            + r"$\mathrm{Orbiting\ Bodies,\ }$"
            + r"$\mathrm{Kinetic\ Energy\ vs.\ Time}$",
            y=1.05,
        )
        ax.set_xlabel(r"$\mathrm{Time}\ \mathrm{(s)}$")
        ax.set_ylabel(r"$\mathrm{Kinetic\ Energy}\ \mathrm{(J)}$")
        for i in range(self.N):
            ax.plot(
                self.t,
                self.ke_tot[:, i, :],
                color=self.colors[list(self.colors.keys())[i]],
                label=self.labels[i],
                alpha=self.alpha,
            )
        ax.legend(loc="lower right")
        outfile = "ke_tot_vs_t"
        outpath = self.outfolder / outfile
        plt.savefig(outpath)

    def plot_pe_tot_vs_t(self) -> None:
        """Plot the total potential energy over time for each of the N bodies."""
        fig = plt.figure(facecolor="white")
        ax = fig.add_subplot(111)
        plt.title(
            "$\mathrm{%s}\ $" % str(self.N)
            + r"$\mathrm{Orbiting\ Bodies,\ }$"
            + r"$\mathrm{Potential\ Energy\ vs.\ Time}$",
            y=1.05,
        )
        ax.set_xlabel(r"$\mathrm{Time}\ \mathrm{(s)}$")
        ax.set_ylabel(r"$\mathrm{Potential\ Energy}\ \mathrm{(J)}$")
        for i in range(self.N):
            ax.plot(
                self.t,
                self.pe_tot[:, i, :],
                color=self.colors[list(self.colors.keys())[i]],
                label=self.labels[i],
                alpha=self.alpha,
            )
        ax.legend(loc="lower right")
        outfile = "pe_tot_vs_t"
        outpath = self.outfolder / outfile
        plt.savefig(outpath)

    def plot_e_sys_vs_t(self) -> None:
        """Plot total kinetic and potential energy of the whole system over time."""
        fig = plt.figure(facecolor="white")
        ax = fig.add_subplot(111)
        plt.title(
            "$\mathrm{%s}\ $" % str(self.N)
            + r"$\mathrm{Orbiting\ Bodies,\ }$"
            + r"$\mathrm{Total\ Energy \ vs.\ Time}$",
            y=1.05,
        )
        ax.set_xlabel(r"$\mathrm{Time (s)}$")
        ax.set_ylabel(r"$\mathrm{Energy (J)}$")
        ax.plot(
            self.t,
            self.ke_sys,
            color="black",
            label=r"$T_{\mathrm{tot}}$",
            alpha=self.alpha,
        )
        ax.plot(
            self.t,
            self.pe_sys,
            color="red",
            label=r"$U_{\mathrm{tot}}$",
            alpha=self.alpha,
        )
        ax.plot(
            self.t,
            self.ke_sys + self.pe_sys,
            color="grey",
            linestyle=":",
            label=r"$T_{\mathrm{tot}}+U_{\mathrm{tot}}$",
            alpha=self.alpha,
        )
        ax.legend(loc="lower right")
        outfile = "e_sys_vs_t"
        outpath = self.outfolder / outfile
        plt.savefig(outpath)

    def plot_phase_space(
        self, qty_name: Literal["p", "ke", "pe"], component: Literal["x", "y", "z"]
    ) -> None:
        """Plot a 3D quantity component against the same position component.

        These types of plots grant insight into the potentially cyclical (i.e. stable)
        nature of an orbital system. For a system where energy is conserved, the areas
        mapped out in phase space should remain constant over time.

        Args:
            qty_name (Literal["p", "ke", "pe"]): One of the 3d quantity arrays:
                momentum `p`, kinetic energy `ke`, or potential energy `pe`.
            component (Literal["x", "y", "z"]): The component of the desired 3D
                quantity to explore.
        """
        if qty_name == "p":
            qty = self.p
            qty_proper_name = r"$\mathrm{Momentum\ }$"
            units = r"$\mathrm{(kg\cdot\frac{m}{s})}$"
        if qty_name == "ke":
            qty = self.ke
            qty_proper_name = r"$\mathrm{Kinetic\ Energy\ }$"
            units = r"$\mathrm{(J)}$"
        if qty_name == "pe":
            qty = self.pe
            qty_proper_name = r"$\mathrm{Potential\ Energy\ }$"
            units = r"$\mathrm{(J)}$"
        component_name = "$\mathrm{%s-}$" % component
        position_proper_name = r"$\mathrm{-Position\ (m)}$"
        fig = plt.figure(facecolor="white")
        ax = fig.add_subplot(111)
        plt.title(
            "$\mathrm{%s}\ $" % str(self.N)
            + r"$\mathrm{Orbiting\ Bodies,\ }$"
            + self.mf_label
            + component_name
            + r"$\mathrm{Component\ }$"
            + qty_proper_name
            + r"$\mathrm{Phase\ Space}$",
            y=1.05,
        )
        ax.set_xlabel(component_name + position_proper_name)
        ax.set_ylabel(component_name + qty_proper_name + units)
        if component == "x":
            ind = 0
        if component == "y":
            ind = 1
        if component == "z":
            ind = 2
        ax.plot(
            self.r[self.lo : self.hi, self.mf, ind],
            qty[self.lo : self.hi, self.mf, ind],
            color=self.colors[list(self.colors.keys())[self.mf]],
            label=self.labels[self.mf],
            alpha=self.alpha,
        )
        outfile = f"{qty_name}_phase_space_{component}_component"
        outpath = self.outfolder / outfile
        plt.savefig(outpath)
