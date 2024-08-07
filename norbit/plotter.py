from typing import Literal

import matplotlib
import matplotlib.animation
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats

from norbit.orbiter import Orbiter

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
        N (int): The number of mutually-interacting orbiting bodies.
        t (np.ndarray): 1D array of ascending time values in steps of `dt`.
        r (np.ndarray): A `num_steps`xNx3 array where each row is a body's position
            (r_x, r_y, r_z).
        p (np.ndarray): A `num_steps`xNx3 array where each row is a body's momentum
            (p_x, p_y, p_z).
        ke (np.ndarray): A `num_steps`xNx3 array where each row is a body's kinetic
            energy (ke_x, ke_y, ke_z).
        ke_tot (np.ndarray): A `num_steps`xNx1 array where each row is a body's total
            kinetic energy (ke_tot_x, ke_tot_y, ke_tot_z).
        ke_sys (np.ndarray): A 1D array of the total kinetic energy of the entire
            system across all time steps.
        pe (np.ndarray): A `num_steps`xNx3 array where each row is a body's potential
            energy (pe_x, pe_y, pe_z).
        pe_tot (np.ndarray): A `num_steps`xNx1 array where each row is a body's total
            potential energy (pe_tot_x, pe_tot_y, pe_tot_z).
        pe_sys (np.ndarray): A 1D array of the total potential energy of the entire
            system across all time steps.
        colors (Dict[str, str]): Labeled HTML color codes corresponding to each body.
            Note that the number of colors must be greater than or equal to the number of bodies.
        mf (int): The chosen body to create phase spots plot for. Think of it as "my
            favorite" mass out of the bunch.
        mf_label (str): The associated label for the chosen body.
        alpha (float): Global transparency value to see where orbits overlap.
        lo (float): Lower bound for arrays used in phase space plots to ensure the
            system is in a "settled down" dynamic equilibrium when plotted.
        hi (float): Upper bound for arrays used in phase space plots to ensure the
            system is in a "settled down" dynamic equilibrium when plotted.
        labels (List[str]): Each mass's LaTeX numerical label (e.g. $m_1$).
        outfolder (Path): The directory where plots will be saved.
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
        self.alpha = 0.7
        self.lo = int(0.35 * (len(self.t)))
        self.hi = int(0.95 * (len(self.t)))
        self.labels = [r"$m_" + str(i) + "$" for i in range(self.N)]
        self.outfolder = solved_orbiter.outfolder
        self.outfolder.mkdir(parents=True, exist_ok=True)

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
        plt.legend(loc="center right", bbox_to_anchor=(0, 0.5))
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
