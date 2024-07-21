import yaml
import numpy as np
from numpy import linalg as LA
from typing import Any, List, Dict, Union
from pathlib import Path
import click


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    with open(config_path, "r") as infile:
        return yaml.safe_load(infile)

def gen_vars(config: Dict[str, Any]):
    for k, v in config.items():
        globals()[k] = v

def accel(m: List, r: np.ndarray, G: float = 6.67e-11) -> np.ndarray:
    """_summary_

    Args:
        m (List): _description_
        r (np.ndarray): _description_
        G (float, optional): _description_. Defaults to 6.67e-11.

    Returns:
        np.ndarray: _description_
    """
    """Calculate the acceleration of each body due to forces from all other bodies.

    Each body's acceleration at each time step is determined by the forces from all
    other bodies. See: https://en.wikipedia.org/wiki/N-body_problem

    Args:
        m (np.ndarray): Nx1 array of the masses of the bodies under consideration.
        r (np.ndarray): An Nx3 array where each row is a body's position (x, y, z).
        G (float, optional): _description_. Defaults to 6.67e-11.

    Returns:
        np.ndarray: An Nx3 array where each row is a body's acceleration (ax, ay, az).
    """
    n = m.shape[0]
    a = np.zeros([n, 3])
    for i in range(n):
        # j is a list of indices of all bodies other than the ith body.
        j = list(range(i)) + list(range(i + 1, n))
        # Each body's acceleration is the sum of n - 1 terms from all other bodies.
        for k in range(n - 1):
            a[i] = G * m[j[k]] / LA.norm(r[i] - r[j[k]]) ** 3 * (r[j[k]] - r[i]) + a[i]
    return a


@click.command()
@click.argument('config_path', default='configs/config.yaml', type=click.Path(exists=True))
def main(config_path: str):
    config = load_config(config_path)
    G = config["G"]
    n = config["n"]
    t0 = config["t0"]
    tf = config["tf"]
    df = config["dt"]
    method = [k for k, v in config["method"].items() if not v][0]

    ic = config["ic"]
    m = config["ics"][ic]["m"][:n]
    r0 = config["ics"][ic]["r0"][:n]
    v0 = config["ics"][ic]["v0"][:n]

    colors = config["colors"]
    figs = config["figs"]
    accel(m, r0, G)

if __name__ == "__main__":
    main()
