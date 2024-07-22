from pathlib import Path
from typing import Any, Dict, Union

import click
import yaml
from orbiter import Orbiter
from plotter import Plotter


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML config file as a `dict`.

    Args:
        config_path (Union[str, Path]): Path to the YAML config file.

    Returns:
        Dict[str, Any]: `dict` containing simulation settings.
    """
    with open(config_path, "r") as infile:
        return yaml.safe_load(infile)


@click.command()
@click.argument(
    "config_path", default="configs/config.yaml", type=click.Path(exists=True)
)
def main(config_path: str) -> None:
    """Run an orbital simulation specified by an associated YAML config file.

    Args:
        config_path (str): Path to the YAML config containing simulation settings.
    """
    config = load_config(config_path)
    print(f"{config["ic"]}, {config["N"]} bodies")
    orbital_system = Orbiter(config)
    if config["run_simulation"]["enabled"]:
        orbital_system.get_orbits()
        orbital_system.get_potential_quantities()
        orbital_system.get_kinetic_quantities()
        if config["run_simulation"]["save_data"]:
            orbital_system.save_quantities()
    if config["load_simulation"]["enabled"]:
        orbital_system.load_quantities(inpath=config["load_simulation"]["inpath"])
    if config["make_plots"]:
        plotter = Plotter(orbital_system)
        if config["plots"]["trajectories"]["3d"]:
            plotter.plot_3d_trajectories()
    print("[END]")


if __name__ == "__main__":
    print("[START]")
    main()
