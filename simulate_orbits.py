import warnings
from pathlib import Path
from typing import Any, Dict, Union

import click
import yaml

from norbit import Animator, Orbiter, Plotter

warnings.filterwarnings("ignore")


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML config file as a `dict`.

    Args:
        config_path (Union[str, Path]): Path to the YAML config file.

    Returns:
        Dict[str, Any]: `dict` containing simulation settings.
    """
    with open(config_path, "r") as infile:
        return yaml.safe_load(infile)


class Simulator:
    def __init__(self, config: Dict[str, Any], orbital_system: Orbiter) -> None:
        """Instantiate a `Simulator` to run an N-body numerical simulation.

        Args:
            config (Dict[str, Any]): Simulation settings for the orbital system.
            orbital_system (Orbiter): A newly instantiated, unsolved orbital system, or
                a solved orbital system, loaded from a PKL file. Note that the config
                specifies whether the orbital system is loaded or not during `Orbiter`
                instantiation.
        """
        self.config = config
        self.orbital_system = orbital_system

    def simulate(self) -> None:
        """Run the orbital N-body simulation according to the provided config."""
        if self.config["run_simulation"]["enabled"]:
            self.orbital_system.get_orbits()
            self.orbital_system.get_potential_qtys()
            self.orbital_system.get_kinetic_qtys()
            if self.config["run_simulation"]["save_data"]:
                self.orbital_system.save()

    def create_visuals(self):
        """Create visuals for a solved orbital system based on the provided config."""
        self.plotter = Plotter(self.orbital_system)
        self.animator = Animator(self.orbital_system)
        all = self.config["plots"]["all"]
        if all or self.config["plots"]["orbits"]["3d"]:
            self.plotter.plot_3d_orbits()
            self.animator.animate_3d_orbits()
        if all or self.config["plots"]["orbits"]["x"]:
            self.plotter.plot_3d_orbits_viewed_from_pos_axis("x")
        if all or self.config["plots"]["orbits"]["y"]:
            self.plotter.plot_3d_orbits_viewed_from_pos_axis("y")
        if all or self.config["plots"]["orbits"]["z"]:
            self.plotter.plot_3d_orbits_viewed_from_pos_axis("z")
        if all or self.config["plots"]["e_sys"]:
            self.plotter.plot_e_sys_vs_t()
        if all or self.config["plots"]["ke_tot"]:
            self.plotter.plot_ke_tot_vs_t()
        if all or self.config["plots"]["pe_tot"]:
            self.plotter.plot_pe_tot_vs_t()
        if all or self.config["plots"]["phase_space"]["p"]["x"]:
            self.plotter.plot_phase_space("p", "x")
        if all or self.config["plots"]["phase_space"]["p"]["y"]:
            self.plotter.plot_phase_space("p", "y")
        if all or self.config["plots"]["phase_space"]["p"]["z"]:
            self.plotter.plot_phase_space("p", "z")
        if all or self.config["plots"]["phase_space"]["pe"]["x"]:
            self.plotter.plot_phase_space("pe", "x")
        if all or self.config["plots"]["phase_space"]["pe"]["y"]:
            self.plotter.plot_phase_space("pe", "y")
        if all or self.config["plots"]["phase_space"]["pe"]["z"]:
            self.plotter.plot_phase_space("pe", "z")
        if all or self.config["plots"]["phase_space"]["ke"]["x"]:
            self.plotter.plot_phase_space("ke", "x")
        if all or self.config["plots"]["phase_space"]["ke"]["y"]:
            self.plotter.plot_phase_space("ke", "y")
        if all or self.config["plots"]["phase_space"]["ke"]["z"]:
            self.plotter.plot_phase_space("ke", "z")


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
    simulator = Simulator(config, orbital_system)
    simulator.simulate()
    simulator.create_visuals()


if __name__ == "__main__":
    main()
