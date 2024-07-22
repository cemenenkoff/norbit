from pathlib import Path
from typing import Any, Dict, Union

import click
import yaml
from orbiter import Orbiter


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
    config = load_config(config_path)
    solar_system = Orbiter(config)
    solar_system.get_orbits()
    solar_system.get_potential_quantities()
    solar_system.get_kinetic_quantities()
    print("end")


if __name__ == "__main__":
    print("start")
    main()
