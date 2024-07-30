<p align="center">
  <img src="images/readme/banner.png", width="800"/>
</p>

# Solving the [N-Body Problem](https://en.wikipedia.org/wiki/N-body_problem) with Numerical Simulations

Conduct gravitational N-body simulations with Norbit, a numerical solution to the [N-body problem](https://en.wikipedia.org/wiki/N-body_problem). Norbit enables you to set up an orbital system with any number of bodies and observe how the system evolves over time.

<p align="center">
  <img src="images/gifs/3d_orbits.gif", width="800"/>
</p>

[TOC]

# 1. Introduction
## 1.1 Background

In physics, the N-body problem is about predicting the motions of celestial objects interacting via gravity. It's essential for understanding the orbits of bodies like the Sun, Moon, and planets. In the 20th century, as astronomers discovered more orbiting bodies in the universe, the desire to solve this problem intensified, but a complete analytical solution remains elusive to this day.

<p align="center">
  <img src="images/gifs/20-3-body-problem-examples-wikipedia.gif", width="800"/>
</p>
<p>
    <em style="font-size: smaller;">These <a href="https://en.wikipedia.org/wiki/Three-body_problem#/media/File:5_4_800_36_downscaled.gif">twenty beautiful examples of the 3-body problem from Wikipedia</a> are an infinitesimally small subset of the N-body solution space.</em>
</p>


## 1.2 High Difficulty
Solving the N-body problem is notoriously difficult because the gravitational interactions between each pair of objects create a highly complex, non-linear system of differential equations that cannot be solved analytically for $N>2$. Additionally, the problem's sensitivity to initial conditions makes long-term predictions highly sensitive to even the smallest perturbations.

<p align="center">
  <img src="images/readme/complete-graphs-by-n.svg", width="800"/>
</p>
<p>
    <em style="font-size: smaller;">An N-body system is an example of a <a href="https://en.wikipedia.org/wiki/Complete_graph">complete graph</a>, a network in which every pair of distinct vertices is connected by a unique edge.</em>
</p>

### 1.2.1 Accounting for Warped Spacetime
Then, to make things even *harder*, a truly complete physical solution needs to include [general relativity](https://en.wikipedia.org/wiki/General_relativity) to account time and space distortions. Despite this, the [two-body problem](https://en.wikipedia.org/wiki/Two-body_problem) and the [three-body problem](https://en.wikipedia.org/wiki/Three-body_problem) (with restrictions) have been fully solved. **Norbit's simulations do not account for warped spacetime.**

# 2. Mathematical Formalism

<p align="center">
  <img src="images/readme/problem-statement-diagram.png", width="400"/>
</p>

## 2.1 Formal Problem Statement
Simply put, the problem is:
  >**Given the current position, velocity, and time of celestial bodies, calculate their gravitational interactions and predict their future motions.**
## 2.2 Strategy
We must solve Newton's [equations of motion](https://en.wikipedia.org/wiki/Equations_of_motion) for N separate bodies in 3D. Given a set of positions, the equation below shows how to obtain the 3D acceleration experienced by body $i$ in the presence of $j$ other bodies.

The accelerations are [numerically integrated](https://en.wikipedia.org/wiki/Numerical_integration) to find velocities, and then the velocities are numerically integrated to find positions.

<p align="center">
  <img src="images/readme/acceleration-equation.png", width="400"/>
</p>

The positions then enable the calculation of potential energies while the velocities corresponding to kinetic energies.

<p align="center">
  <img src="images/readme/pe-equation.png", width="400"/>
</p>

## 2.3 Multidimensional Arrays
Norbit approaches this problem using [matrix](https://en.wikipedia.org/wiki/Matrix_(mathematics)) algebra, enabled by multidimensional arrays, and facilitated by [`pandas`](https://pandas.pydata.org/) and [`numpy`](https://numpy.org/). Think of each moment in time as a layer in a large stack. Each moment is a snapshot of six critical numbers associated with each orbiting body, exactly at that moment: $x$-position, $y$-position, $z$-position, $x$-velocity, $y$-velocity, and $z$-velocity.

<p align="center">
  <img src="images/readme/multidimensional-array.png", width="400"/>
</p>

## 2.3 Numerical Implementation
We need to *step* the simulation through time, and the lower the step size, the more accurate the simulation. Step size is not the only important factor though. Using different stepping *methods* can drastically improve the precision of the orbital trajectory calculations. The methods Norbit employs are listed here in ascending order of precision. Refer to `orbiter.Orbiter` to review the actual algorithms.
  1. [Euler](https://en.wikipedia.org/wiki/Euler_method)
  2. [Euler-Cromer](https://en.wikipedia.org/wiki/Semi-implicit_Euler_method)
  3. [Second-Order Runge-Kutta](https://math.libretexts.org/Workbench/Numerical_Methods_with_Applications_(Kaw)/8%3A_Ordinary_Differential_Equations/8.03%3A_Runge-Kutta_2nd-Order_Method_for_Solving_Ordinary_Differential_Equations)
  4. [Fourth-Order Runge-Kutta](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods)
  5. [Velocity Verlet](https://en.wikipedia.org/wiki/Verlet_integration)
  6. [Position Verlet](https://young.physics.ucsc.edu/115/leapfrog.pdf)
  7. [Velocity Extended Forest-Ruth-Like](https://arxiv.org/pdf/cond-mat/0110585)
  8. [Position Extended Forest-Ruth-Like](https://en.wikipedia.org/wiki/Symplectic_integrator)



# 3. Core Files
## 3.1 [`simulate_orbits.py`](simulate_orbits.py)
Run `python simulate_orbits.py` to interface with this project. `config.yaml` directs this script on whether to run a new simulation, load an old one, plot results, or transform results into an animated GIF.

## 3.2 [`config.yaml`](configs/config.yaml)
Define how the script should run. Many of the options are self-explanatory, but if you need clarification, refer to the `Simulator` class in `simulate_orbits.py`. Look to `config.yaml` for a concrete example.

## 3.3 [`animator.py`](norbit/animator.py)
Animate 3D orbital trajectories with the `Animator` class, given a solved orbital system.

## 3.4 [`orbiter.py`](norbit/orbiter.py)
Calculate (i.e. solve for) 3D orbital trajectories with the `Orbiter` class.

## 3.5 [`plotter.py`](norbit/plotter.py)
Plot 3D orbital trajectories with the `Plotter` class, given a solved orbital system.

# 4. 10-Body Investigation
Here we compare the results of a 25-year simulation and a 500-year simulation of a stable orbital system similar to our solar system. Both sims used the Euler method, and because of the method's in-built imprecision, the orbits "smeared" as time goes on. In a stable system (i.e. one where the total energy is negative), this smearing indicates energy loss in the system over time.

## 4.2 25-Year Euler Simulation
<p align="center">
  <img src="images/readme/10-orbiting-bodies.png", width="400"/>
</p>

<p align="center">
  <img src="images/readme/10-orbiting-bodies-tot-sys-pe-ke.png", width="400"/>
</p>

## 4.3 500-Year Euler Simulation
<p align="center">
  <img src="images/readme/10-orbiting-bodies-500-yrs.png", width="400"/>
</p>

<p align="center">
  <img src="images/readme/10-orbiting-bodies-tot-sys-pe-ke-500-yrs.png", width="400"/>
</p>

# 5. Future Roadmap
Here are some ideas for future development.
- Enable animation of 2D projections.
- Enable animation of all graphs.
- Enable multithreading or parallelization to speed up calculations.
- Create a dynamic connection to the [NASA JPL Horizon System](https://ssd.jpl.nasa.gov/horizons/app.html#/) to get the most up-to-date planetary data of our solar system.

## 6. Appendix: Setup for New Developers
If you are fairly new to Python programming, I'd recommend setting up this project by following these steps. If you want more in depth knowledge about environment setup, I'd recommend you read [my tutorial on interfacing with the computer like a software developer](https://github.com/cemenenkoff/python-essentials-for-stem-wizards).

1. Download and install [VS Code](https://code.visualstudio.com/download).

2. Install [Python 3.12.4](https://www.python.org/downloads/release/python-3124/) (☑️ **Add python.exe to PATH** if you have no other Python versions installed).

3. Install [Git bash](https://git-scm.com/downloads).

4. Open VS Code.

5. Press `F1`, and in the command palette, search for `Terminal: Select Default Profile` and set Git bash as the default terminal.

6. Start a new terminal with `Ctrl` + `` ` ``.

7. Clone this repository to a directory where you like to store your coding projects.

8. Open this repository (i.e. the `norbit` folder) as the current workspace folder with `Ctrl` + `K` `Ctrl` + `O`.

9.  Make sure the terminal path points to the `norbit` folder, and if it doesn't, navigate there via `cd <path_to_norbit_folder>`. You can confirm you're in the right spot with quick `ls -la` command.

10. From the terminal, run `pip install virtualenv` to install the `virtualenv` module.

11. Run `python -m virtualenv <myenvname> --python=python3.12.4` to create a virtual environment that runs on Python 3.12.4.

12. Activate the virtual environment with `source <myenvname>/Scripts/activate`.

13. You should see `(<myenvname>)` two lines above the terminal input line when the environment is active.

14. Press `F1` to open VS Code's command palette, then search for `Python: Select Interpreter` and select `Python 3.12.4 64-bit ('<myenvname>':venv)`.

15. Run `pip install -r requirements.txt` to install all dependencies on your activated virtual environment.

16. Run `simulate_orbits.py`.