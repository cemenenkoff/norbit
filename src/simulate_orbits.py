import warnings
from pathlib import Path
from typing import Any, Dict, Union

import click
import yaml
from orbiter import Orbiter
from plotter import Plotter

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
        orbital_system.get_potential_qtys()
        orbital_system.get_kinetic_qtys()
        if config["run_simulation"]["save_data"]:
            orbital_system.save()
    if config["load_simulation"]["enabled"]:
        inpath = Path(config["load_simulation"]["inpath"])
        orbital_system.load(inpath)
    if config["make_plots"]:
        plotter = Plotter(orbital_system)
        if config["plots"]["all"] or config["plots"]["orbits"]["3d"]:
            plotter.plot_3d_orbits()
        if config["plots"]["all"] or config["plots"]["orbits"]["x"]:
            plotter.plot_3d_orbits_viewed_from_pos_axis("x")
        if config["plots"]["all"] or config["plots"]["orbits"]["y"]:
            plotter.plot_3d_orbits_viewed_from_pos_axis("y")
        if config["plots"]["all"] or config["plots"]["orbits"]["z"]:
            plotter.plot_3d_orbits_viewed_from_pos_axis("z")
        if config["plots"]["all"] or config["plots"]["e_sys"]:
            plotter.plot_e_sys_vs_t()
        if config["plots"]["all"] or config["plots"]["ke_tot"]:
            plotter.plot_ke_tot_vs_t()
        if config["plots"]["all"] or config["plots"]["pe_tot"]:
            plotter.plot_pe_tot_vs_t()
        if config["plots"]["all"] or config["plots"]["phase_space"]["p"]["x"]:
            plotter.plot_phase_space("p", "x")
        if config["plots"]["all"] or config["plots"]["phase_space"]["p"]["y"]:
            plotter.plot_phase_space("p", "y")
        if config["plots"]["all"] or config["plots"]["phase_space"]["p"]["z"]:
            plotter.plot_phase_space("p", "z")
        if config["plots"]["all"] or config["plots"]["phase_space"]["pe"]["x"]:
            plotter.plot_phase_space("pe", "x")
        if config["plots"]["all"] or config["plots"]["phase_space"]["pe"]["y"]:
            plotter.plot_phase_space("pe", "y")
        if config["plots"]["all"] or config["plots"]["phase_space"]["pe"]["z"]:
            plotter.plot_phase_space("pe", "z")
        if config["plots"]["all"] or config["plots"]["phase_space"]["ke"]["x"]:
            plotter.plot_phase_space("ke", "x")
        if config["plots"]["all"] or config["plots"]["phase_space"]["ke"]["y"]:
            plotter.plot_phase_space("ke", "y")
        if config["plots"]["all"] or config["plots"]["phase_space"]["ke"]["z"]:
            plotter.plot_phase_space("ke", "z")
    print("[END]")


if __name__ == "__main__":
    print("[START]")
    main()
