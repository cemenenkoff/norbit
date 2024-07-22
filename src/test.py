import yaml
import numpy as np
from numpy import linalg as LA
from typing import Any, Dict, Union, Tuple
from pathlib import Path
import click
from tqdm import tqdm


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML config file as a `dict`.

    Args:
        config_path (Union[str, Path]): Path to the YAML config file.

    Returns:
        Dict[str, Any]: `dict` containing simulation settings.
    """
    with open(config_path, "r") as infile:
        return yaml.safe_load(infile)


def convert_times_get_steps(
    t0: float, tf: float, dt: float
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


def get_accel(m: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Get the acceleration of each body due to forces from all other bodies.

    Each body's acceleration at each time step is determined by the forces from all
    other bodies. See: https://en.wikipedia.org/wiki/N-body_problem

    Args:
        m (np.ndarray): A 1D array of the masses of the bodies under consideration.
        r (np.ndarray): A `num_steps`xNx3 array where each row is a body's position
            (x, y, z).

    Returns:
        np.ndarray: A `num_steps`xNx3 array where each row is a body's acceleration
            (ax, ay, az).
    """
    n = len(m)
    a = np.zeros([n, 3])
    for i in range(n):
        j = list(range(i)) + list(range(i + 1, n))  # All bodies other than the ith.
        # Each body's acceleration is the sum of the n - 1 forces from the other bodies.
        for k in range(n - 1):
            a[i] = G * m[j[k]] / LA.norm(r[i] - r[j[k]]) ** 3 * (r[j[k]] - r[i]) + a[i]
    return a


def get_orbits(
    m: np.ndarray,
    r0: np.ndarray,
    v0: np.ndarray,
    num_steps: int,
    dt: float,
    method: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the trajectories of N gravitationally interacting bodies.

    Args:
        m (np.ndarray): A 1D array of the masses of the bodies under consideration.
        r0 (np.ndarray): An Nx3 array where each row is a body's initial position
            (x0, y0, z0)
        v0 (np.ndarray): An Nx3 array where each row is a body's initial velocity
            (v0, v0, v0)
        num_steps (int): The number of time steps of length `dt` to simulate.
        dt (float): The time step (in seconds) between each recalculation.
        method (str): The chosen numerical integration stepping method.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The 3D position data over time and the 3D
            velocity data over time, each as separate `num_steps`xNx3 arrays.
    """
    # If we print r or v below, we'll see several "layers" of Nx3 matrices. Each layer # represents a moment in time (i.e. one time step). Within each Nx3 matrix, the row
    # denotes which body, while the columns 1-3 (indexed 0-2 in Python) represent x-, #
    # y-, z-components of the quantity respectively. In essence, each number in r or v
    # is associated with three indices: time step, body, and component.
    r = np.zeros([num_steps + 1, N, 3])
    v = np.zeros([num_steps + 1, N, 3])
    r[0] = r0
    v[0] = v0

    # Numerically integrate accelerations into velocities and then positions.
    for i in tqdm(range(num_steps)):
        # The simplest method is the Euler method. It does not conserve energy.
        if method == "euler":
            r[i + 1] = r[i] + dt * v[i]
            v[i + 1] = v[i] + dt * get_accel(m, r[i])
        # The Euler-Cromer method drives our next-simplest stepper.
        if method == "ec":
            r[i + 1] = r[i] + dt * v[i]
            v[i + 1] = v[i] + dt * get_accel(m, r[i + 1])

        # Getting slightly fancier, we employ the 2nd Order Runge-Kutta method.
        if method == "rk2":
            v_iphalf = v[i] + get_accel(m, r[i]) * (dt / 2)  # Think like v[i + 0.5].
            r_iphalf = r[i] + v[i] * (dt / 2)
            v[i + 1] = v[i] + get_accel(m, r_iphalf) * dt
            r[i + 1] = r[i] + v_iphalf * dt
        # Even fancier, here's the 4th Order Runge-Kutta method.
        if method == "rk4":
            r1 = r[i]
            v1 = v[i]
            a1 = get_accel(m, r1)
            r2 = r1 + (dt / 2) * v1
            v2 = v1 + (dt / 2) * a1
            a2 = get_accel(m, r2)
            r3 = r1 + (dt / 2) * v2
            v3 = v1 + (dt / 2) * a2
            a3 = get_accel(m, r3)
            r4 = r1 + dt * v3
            v4 = v1 + dt * a3
            a4 = get_accel(m, r4)
            r[i + 1] = r[i] + (dt / 6) * (v1 + 2 * v2 + 2 * v3 + v4)
            v[i + 1] = v[i] + (dt / 6) * (a1 + 2 * a2 + 2 * a3 + a4)

        # Velocity Verlet implementation. See: https://tinyurl.com/bdffjnh9
        if method == "vv":
            v_iphalf = v[i] + (dt / 2) * get_accel(r[i])
            r[i + 1] = r[i] + dt * v_iphalf
            v[i + 1] = v_iphalf + (dt / 2) * get_accel(r[i + 1])
        # Position Verlet implementation (found in the same pdf as "vv").
        if method == "pv":
            r_iphalf = r[i] + (dt / 2) * v[i]
            v[i + 1] = v[i] + dt * get_accel(r_iphalf)
            r[i + 1] = r_iphalf + (dt / 2) * v[i + 1]

        # EFRL refers to an extended Forest-Ruth-like integration algorithm.
        opt_e = 0.1786178958448091e0
        opt_l = -0.2123418310626054e0
        opt_k = -0.6626458266981849e-1
        # Below is a velocity EFRL implementation (VEFRL).
        # See: https://arxiv.org/pdf/cond-mat/0110585.pdf
        # Optimization parameters associated with EFRL routines.
        if method == "vefrl":
            v1 = v[i] + get_accel(r[i]) * opt_e * dt
            r1 = r[i] + v1 * (1 - 2 * opt_l) * (dt / 2)
            v2 = v1 + get_accel(r1) * opt_k * dt
            r2 = r1 + v2 * opt_l * dt
            v3 = v2 + get_accel(r2) * (1 - 2 * (opt_k + opt_e)) * dt
            r3 = r2 + v3 * opt_l * dt
            v4 = v3 + get_accel(r3) * opt_k * dt
            r[i + 1] = r3 + v4 * (1 - 2 * opt_l) * (dt / 2)
            v[i + 1] = v4 + get_accel(r[i + 1]) * opt_e * dt
        # Position EFRL (PEFRL) (found in the same pdf as "vefrl").
        if method == "pefrl":
            r1 = r[i] + v[i] * opt_e * dt
            v1 = v[i] + get_accel(r1) * (1 - 2 * opt_l) * (dt / 2)
            r2 = r1 + v1 * opt_k * dt
            v2 = v1 + get_accel(r2) * opt_l * dt
            r3 = r2 + v2 * (1 - 2 * (opt_k + opt_e)) * dt
            v3 = v2 + get_accel(r3) * opt_l * dt
            r4 = r3 + v3 * opt_k * dt
            v[i + 1] = v3 + get_accel(r4) * (1 - 2 * opt_l) * (dt / 2)
            r[i + 1] = r4 + v[i + 1] * opt_e * dt
    return r, v


def get_pot_energy_sys(m: np.ndarray, r: np.ndarray) -> float:
    """Get the total gravitational potential energy for n bodies given their positions.

    The nested for loop iterates over pairs of bodies only once by ensuring that i < j,
    thus preventing double-counting of pairwise interactions and ensuring that each
    pair is considered only once.

    Args:
        m (np.ndarray): A 1D array of the masses of the bodies under consideration.
        r (np.ndarray): A `num_steps`xNx3 array where each row is a body's position
            (x, y, z).

    Returns:
        float: Total gravitational potential energy of the system (float).
    """
    u_sys = 0
    for j in range(N):
        for i in range(j):
            u_sys += -G * m[i] * m[j] / LA.norm(r[i] - r[j])
    return u_sys


def get_kinetic_quantities(
    num_steps: isinstance, m: np.ndarray, v: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get an orbiting system's kinetic energy and momentum from body velocities.

    Args:
        num_steps (int): The number of time steps of length `dt` to simulate.
        m (np.ndarray): A 1D array of the masses of the bodies under consideration.
        v (np.ndarray): A `num_steps`xNx3 array where each row is a body's velocity
            (vx, vy, vz).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The 3D kinetic energy of the system (over time)
    """
    n = len(m)
    ke = np.zeros((num_steps, n, 3))  # 3D kinetic energy data for each body.
    p = np.zeros((num_steps, n, 3))  # 3D momentum data for each body.
    ke_tot = np.zeros((num_steps, n, 1))  # Total kinetic energy for each body.
    ke_tot_sys = np.zeros((num_steps, 1))  # Total kinetic energy of the system.
    v2 = v**2  # Square all velocities for use in the energy calculation.
    for i in tqdm(range(num_steps)):
        for j in range(n):
            ke[i, j, :] = (m[j] / 2) * v2[i, j, :]
            ke_tot[i, j, 0] = sum(ke[i, j, :])
            p[i, j, :] = m[j] * v[i, j, :]
        ke_tot_sys[i] = sum(ke_tot[i])
    return ke, p, ke_tot, ke_tot_sys


@click.command()
@click.argument(
    "config_path", default="configs/config.yaml", type=click.Path(exists=True)
)
def main(config_path: str):
    config = load_config(config_path)
    method = [k for k, v in config["method"].items() if v][0]

    global G, N
    G = config["G"]
    N = config["N"]

    t0, tf, dt, num_steps = convert_times_get_steps(
        config["t0"], config["tf"], config["dt"]
    )
    t = dt * np.array(range(num_steps))  # Ascending time values.

    ic = config["ic"]
    # Trim the initial conditions arrays to represent N bodies.
    m = np.array(config["ics"][ic]["m"][:N])
    r0 = np.array(config["ics"][ic]["r0"][:N])
    v0 = np.array(config["ics"][ic]["v0"][:N])

    r, v = get_orbits(m, r0, v0, num_steps, dt, method)
    u_sys = get_pot_energy_sys(m, r)
    ke, p, ke_tot, ke_tot_sys = get_kinetic_quantities(num_steps, m, v)

    colors = config["colors"]
    figs = config["figs"]
    print("end")


if __name__ == "__main__":
    print("start")
    main()
