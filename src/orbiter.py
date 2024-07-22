import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from numpy import linalg as LA
from tqdm import tqdm

G = 6.67e-11  # Universal gravitational constant in units of m^3/kg*s^2.


class Orbiter:
    """Explore a 3D N-body system with numerical simulation.

    `Orbiter` uses numerical integration to simulate N mutually-interacting point
    masses, each one exerting a dynamically changing amount gravitational force on each
    of the N - 1 others. Mathematics cannot yet analytically solve the N-body problem,
    so it can realistically only be explored through numerical simulation for
    reasonably large N.

    Attributes:
        method (str): The chosen numerical integration stepping method.
        N (int): The number of mutually-interacting orbiting bodies.
        t0 (float): Start time in seconds.
        tf (float): End time in seconds.
        dt (float): The time step (in seconds) between each recalculation.
        num_steps (int): The number of time steps of length `dt` to simulate.
        t (np.ndarray): 1D array of ascending time values in steps of `dt`.
        ic (str): The name of the initial condition configuration.
        name (str): Alias for `ic`.
        m (np.ndarray): A 1D array of the masses of the bodies under consideration.
        r0 (np.ndarray): An Nx3 array where each row is a body's initial position
            (x0, y0, z0).
        v0 (np.ndarray): An Nx3 array where each row is a body's initial velocity
            (v0, v0, v0).
        runfolder (str): __description__.
        outfolder (Path): __description__.
        _a (np.ndarray): __description__.
        _r (np.ndarray): __description__.
        _v (np.ndarray): __description__.
        _p (np.ndarray): __description__.
        _pe (np.ndarray): __description__.
        _pe_tot (np.ndarray): __description__.
        _pe_sys (np.ndarray): __description__.
        _ke (np.ndarray): __description__.
        _ke_tot (np.ndarray): __description__.
        _ke_sys (np.ndarray): __description__.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Instantiate an `Orbiter` simulation object.

        Args:
            config (Dict[str, Any]): Configuration dictionary with simulation settings.
        """
        self.colors = config["colors"]
        self.method = [k for k, v in config["method"].items() if v][0]
        self.mf = config["mf"]
        self.N = config["N"]
        self.t0, self.tf, self.dt, self.num_steps = self.convert_times_get_steps(
            config["t0"], config["tf"], config["dt"]
        )
        self.t = self.dt * np.array(range(self.num_steps))
        self.ic = config["ic"]
        self.name = self.ic
        # Trim the initial conditions arrays to represent N bodies.
        self.m = np.array(config["ics"][self.ic]["m"][: self.N])
        self.r0 = np.array(config["ics"][self.ic]["r0"][: self.N])
        self.v0 = np.array(config["ics"][self.ic]["v0"][: self.N])
        self.runfolder = (
            f"{self.method}_{config['t0']}t0_{config['tf']}tf_{config['dt']}dt"
        )
        self.outfolder = (
            Path.cwd() / "runs" / self.name / f"{self.N}_bodies" / self.runfolder
        )
        self.outfolder.mkdir(parents=True, exist_ok=True)
        self._a = None
        self._r = None
        self._v = None
        self._p = None
        self._pe = None
        self._pe_tot = None
        self._pe_sys = None
        self._ke = None
        self._ke_tot = None
        self._ke_sys = None

    @property
    def a(self):
        return self._a

    @property
    def r(self):
        return self._r

    @property
    def v(self):
        return self._v

    @property
    def p(self):
        return self._p

    @property
    def pe(self):
        return self._pe

    @property
    def pe_tot(self):
        return self._pe_tot

    @property
    def pe_sys(self):
        return self._pe_sys

    @property
    def ke(self):
        return self._ke

    @property
    def ke_tot(self):
        return self._ke_tot

    @property
    def ke_sys(self):
        return self._ke_sys

    def convert_times_get_steps(
        self, t0: float, tf: float, dt: float
    ) -> Tuple[float, float, float, int]:
        """Get relevant time values for orbital simulations.

        Args:
            t0 (float): Start time in years.
            tf (float): End time in years.
            dt (float): Time step in hours.

        Returns:
            Tuple[int, float, float, float]: The start time, end time, time step, all converted to seconds, and then the resulting number of steps.
        """
        t0 = t0 * 365.26 * 24 * 3600  # Convert t0 from years to seconds.
        tf = tf * 365.26 * 24 * 3600  # Convert tf from years to seconds.
        dt = dt * 3600.0  # Convert dt from hours to seconds.
        num_steps = int(abs(tf - t0) / dt)  # Define the total number of time steps.
        # t = dt * np.array(range(num_steps))  # Ascending time values.
        return t0, tf, dt, num_steps

    def get_accel(self, r: np.ndarray) -> np.ndarray:
        """Get the acceleration of each body due to forces from all other bodies.

        Each body's acceleration at each time step is determined by the forces from all
        other bodies. See: https://en.wikipedia.org/wiki/N-body_problem

        Args:
            r (np.ndarray): A `num_steps`xNx3 array where each row is a body's position
                (x, y, z).

        Returns:
            np.ndarray: A `num_steps`xNx3 array where each row is a body's acceleration
                (ax, ay, az).
        """
        self.N
        a = np.zeros([self.N, 3])
        for i in range(self.N):
            j = list(range(i)) + list(
                range(i + 1, self.N)
            )  # All bodies other than the ith.
            # Each body's acceleration is the sum of the n - 1 forces from the other bodies.
            for k in range(self.N - 1):
                a[i] = (
                    G * self.m[j[k]] / LA.norm(r[i] - r[j[k]]) ** 3 * (r[j[k]] - r[i])
                    + a[i]
                )
        self._a = a
        return a

    def get_orbits(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the trajectories of N gravitationally interacting bodies.

        Note that this method updates the `r` and `v` attributes.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - 3D position data `r` (`num_steps`xNx3).
                - 3D velocity data `v` (`num_steps`xNx3).
        """
        # If we print r or v below, we'll see several "layers" of Nx3 matrices. Each
        # layer represents a moment in time (i.e. one time step). Within each Nx3
        # matrix, the row denotes which body, while the columns 1-3 (indexed 0-2 in
        # Python) represent x-, y-, z-components of the quantity respectively. In
        # essence, each number in r or v is associated with three indices: time step,
        # body, and component.
        r = np.zeros([self.num_steps + 1, self.N, 3])
        v = np.zeros([self.num_steps + 1, self.N, 3])
        r[0] = self.r0
        v[0] = self.v0
        # Numerically integrate accelerations into velocities and then positions.
        for i in tqdm(range(self.num_steps)):
            # The simplest method is the Euler method. It does not conserve energy.
            if self.method == "euler":
                r[i + 1] = r[i] + self.dt * v[i]
                v[i + 1] = v[i] + self.dt * self.get_accel(r[i])
            # The Euler-Cromer method drives our next-simplest stepper.
            if self.method == "ec":
                r[i + 1] = r[i] + self.dt * v[i]
                v[i + 1] = v[i] + self.dt * self.get_accel(r[i + 1])
            # Getting slightly fancier, we employ the 2nd Order Runge-Kutta method.
            if self.method == "rk2":
                v_iphalf = v[i] + self.get_accel(r[i]) * (
                    self.dt / 2
                )  # Think like v[i + 0.5].
                r_iphalf = r[i] + v[i] * (self.dt / 2)
                v[i + 1] = v[i] + self.get_accel(r_iphalf) * self.dt
                r[i + 1] = r[i] + v_iphalf * self.dt
            # Even fancier, here's the 4th Order Runge-Kutta method.
            if self.method == "rk4":
                r1 = r[i]
                v1 = v[i]
                a1 = self.get_accel(r1)
                r2 = r1 + (self.dt / 2) * v1
                v2 = v1 + (self.dt / 2) * a1
                a2 = self.get_accel(r2)
                r3 = r1 + (self.dt / 2) * v2
                v3 = v1 + (self.dt / 2) * a2
                a3 = self.get_accel(r3)
                r4 = r1 + self.dt * v3
                v4 = v1 + self.dt * a3
                a4 = self.get_accel(r4)
                r[i + 1] = r[i] + (self.dt / 6) * (v1 + 2 * v2 + 2 * v3 + v4)
                v[i + 1] = v[i] + (self.dt / 6) * (a1 + 2 * a2 + 2 * a3 + a4)
            # Velocity Verlet implementation. See: https://tinyurl.com/bdffjnh9
            if self.method == "vv":
                v_iphalf = v[i] + (self.dt / 2) * self.get_accel(r[i])
                r[i + 1] = r[i] + self.dt * v_iphalf
                v[i + 1] = v_iphalf + (self.dt / 2) * self.get_accel(r[i + 1])
            # Position Verlet implementation (found in the same pdf as "vv").
            if self.method == "pv":
                r_iphalf = r[i] + (self.dt / 2) * v[i]
                v[i + 1] = v[i] + self.dt * self.get_accel(r_iphalf)
                r[i + 1] = r_iphalf + (self.dt / 2) * v[i + 1]
            # EFRL refers to an extended Forest-Ruth-like integration algorithm.
            opt_e = 0.1786178958448091e0
            opt_l = -0.2123418310626054e0
            opt_k = -0.6626458266981849e-1
            # Below is a velocity EFRL implementation (VEFRL).
            # See: https://arxiv.org/pdf/cond-mat/0110585.pdf
            # Optimization parameters associated with EFRL routines.
            if self.method == "vefrl":
                v1 = v[i] + self.get_accel(r[i]) * opt_e * self.dt
                r1 = r[i] + v1 * (1 - 2 * opt_l) * (self.dt / 2)
                v2 = v1 + self.get_accel(r1) * opt_k * self.dt
                r2 = r1 + v2 * opt_l * self.dt
                v3 = v2 + self.get_accel(r2) * (1 - 2 * (opt_k + opt_e)) * self.dt
                r3 = r2 + v3 * opt_l * self.dt
                v4 = v3 + self.get_accel(r3) * opt_k * self.dt
                r[i + 1] = r3 + v4 * (1 - 2 * opt_l) * (self.dt / 2)
                v[i + 1] = v4 + self.get_accel(r[i + 1]) * opt_e * self.dt
            # Position EFRL (PEFRL) (found in the same pdf as "vefrl").
            if self.method == "pefrl":
                r1 = r[i] + v[i] * opt_e * self.dt
                v1 = v[i] + self.get_accel(r1) * (1 - 2 * opt_l) * (self.dt / 2)
                r2 = r1 + v1 * opt_k * self.dt
                v2 = v1 + self.get_accel(r2) * opt_l * self.dt
                r3 = r2 + v2 * (1 - 2 * (opt_k + opt_e)) * self.dt
                v3 = v2 + self.get_accel(r3) * opt_l * self.dt
                r4 = r3 + v3 * opt_k * self.dt
                v[i + 1] = v3 + self.get_accel(r4) * (1 - 2 * opt_l) * (self.dt / 2)
                r[i + 1] = r4 + v[i + 1] * opt_e * self.dt
        self._r = r
        self._v = v
        return r, v

    def get_potential_qtys(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get an orbiting system's potential energy from body positions.

        Note that `get_orbits` must be called at least once before this method is
        called so that `r` is appropriately calculated (and is not None).

        This method updates the `pe`, `pe_tot`, and `pe_sys` attributes.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - 3D potential energy `pe` (`num_steps`xNx3).
                - 3D potential energy `pe_tot` for each body (`num_steps`xNx1).
                - Total potential energy of the system `pe_sys` (`num_steps`x1).
        """
        self.N = len(self.m)
        pe = np.zeros((self.num_steps, self.N, 3))  # 3D potential energy for each body.
        pe_tot = np.zeros((self.num_steps, self.N, 1))  # Tot. pot. energy, each body.
        pe_sys = np.zeros((self.num_steps, 1))  # Total potential energy, system-wide.
        for i in tqdm(range(self.num_steps)):
            for j in range(self.N):
                for k in range(j):
                    r_diff = self.r[i, j, :] - self.r[i, k, :]
                    r_mag = LA.norm(r_diff)
                    # Avoid division by zero if r_mag is zero (which shouldn't happen
                    # in a well-defined system).
                    if r_mag > 0:
                        potential = -G * self.m[j] * self.m[k] / r_mag
                        pe[i, j, :] += potential * (
                            r_diff / r_mag
                        )  # Distribute potential along the vector
                        pe_tot[i, j, 0] += potential
            pe_sys[i] = sum(pe_tot[i].flatten())
        self._pe = pe
        self._pe_tot = pe_tot
        self._pe_sys = pe_sys
        return pe, pe_tot, pe_sys

    def get_kinetic_qtys(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get an orbiting system's kinetic energy and momentum from body velocities.

        Note that `get_orbits` must be called at least once before this method is
        called so that `v` is appropriately calculated (and is not None).

        This method updates the `ke`, `p`, `ke_tot`, and `ke_sys` attributes.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - 3D momentum `p` (`num_steps`xNx3).
                - 3D kinetic energy `ke` (`num_steps`xNx3).
                - Total kinetic energy of each body `ke_tot` (`num_steps`xNx1).
                - Total kinetic energy of the system `ke_sys` (`num_steps`x1).
        """
        self.N = len(self.m)
        ke = np.zeros((self.num_steps, self.N, 3))  # 3D kinetic energy for each body.
        p = np.zeros((self.num_steps, self.N, 3))  # 3D momentum for each body.
        ke_tot = np.zeros((self.num_steps, self.N, 1))  # Tot kinetic energy, each body.
        ke_sys = np.zeros((self.num_steps, 1))  # Total kinetic energy, system-wide.
        v2 = self.v**2  # Squared velocities.
        for i in tqdm(range(self.num_steps)):
            for j in range(self.N):
                p[i, j, :] = self.m[j] * self.v[i, j, :]
                ke[i, j, :] = (self.m[j] / 2) * v2[i, j, :]
                ke_tot[i, j, 0] = sum(ke[i, j, :])
            ke_sys[i] = sum(ke_tot[i])
        self._p = p
        self._ke = ke
        self._ke_tot = ke_tot
        self._ke_sys = ke_sys
        return p, ke, ke_tot, ke_sys

    def save(self) -> None:
        """Pickle a snapshot of an `Orbiter` object's attributes."""
        filename = "orbital_data.pkl"
        outpath = self.outfolder / filename
        with outpath.open("wb") as outfile:
            pickle.dump(self.__dict__, outfile, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, inpath: Path) -> None:
        """Load `Orbiter` attributes from a pickled snapshot (i.e. PKL file).

        Args:
            inpath (Path, optional): Path to the PKL file.
        """
        with inpath.open("rb") as infile:
            attributes = pickle.load(infile)
        for k, v in attributes.items():
            setattr(self, k, v)
