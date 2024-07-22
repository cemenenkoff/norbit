# norbit
Explore gravitational N-body simulations.

# Background
In the final quarter of my physics degree, for a computational physics class project, we had the opportunity to select a physical process to simulate using a computer. I chose to simulate the N-body problem in Python, and this repository was born.

# Config Attributes:
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