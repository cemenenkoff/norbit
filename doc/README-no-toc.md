# norbit
Explore gravitational N-body simulations.

<p align="center">
  <img src="img/gifs/3d_orbits.gif", width="400"/>
</p>

[TOC]

# 1. Introduction
The N-body problem involves predicting the individual motions of a group of celestial objects interacting with each other gravitationally. Given the initial positions, velocities, and masses of these objects, the challenge is to solve their equations of motion to understand their future positions and velocities over time.

Solving the N-body problem is notoriously difficult because the gravitational interactions between each pair of objects create a highly complex, non-linear system of differential equations that cannot be solved analytically for $N>2$. Additionally, the problem's sensitivity to initial conditions, known as chaos, makes long-term predictions highly sensitive to even the smallest perturbations.

# 2. Config Attributes:
    method (str): The chosen numerical integration stepping method.
    N (int): The number of mutually-interacting orbiting bodies under consideration.
    t0 (float): Start time in seconds.
    tf (float): End time in seconds.
    dt (float): The time step (in seconds) between each recalculation.
    num_steps (int): The number of time steps of length `dt` to simulate.
    t (np.ndarray): 1D array of ascending time values in steps of `dt`.
    ic (str): The name of the initial condition configuration.
    name (str): Alias for `ic`.
    m (np.ndarray): A 1D array of the masses of the bodies under consideration.
    r0 (np.ndarray): An Nx3 array where each row is a body's initial position
        (x0, y0, z0)
    v0 (np.ndarray): An Nx3 array where each row is a body's initial velocity
        (v0, v0, v0)