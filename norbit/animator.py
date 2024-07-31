import re
from typing import List, Tuple

import click
import matplotlib as mpl
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

if __name__ == "__main__":
    from orbiter import Orbiter
else:
    from norbit.orbiter import Orbiter
from tqdm import tqdm


class Animator:
    """Animate 3D orbital trajectories.

    Attributes:
        solved_orbiter (Orbiter): An orbital system with each body's trajectory fully
            mapped out over the give time span.
    """

    def __init__(self, solved_orbiter: Orbiter) -> None:
        """Instantiate an `Animator` to create an animation of an orbital system.

        Args:
            solved_orbiter (Orbiter): An `Orbiter` object that has been fully solved.
        """
        self.solved_orbiter = solved_orbiter

    def update_graph(
        self,
        time_step: int,
        dfs: List[pd.DataFrame],
        lines: List[Line2D],
        title: mpl.text.Text,
        pbar: tqdm,
    ) -> Tuple[mpl.text.Text, plt.Figure]:
        """
        Update the plot for each time step.

        The `update_graph` function returns objects that need to be re-drawn with each
        frame update in the animation. In the context of `FuncAnimation`, this is done to
        optimize the rendering process. Returning the objects that change allows
        `FuncAnimation` to only update those specific parts of the plot, which can
        significantly improve performance, especially for complex animations.

        Args:
            time_step (int): The current time step in the animation.
            dfs (List[pd.DataFrame]): A list of (t, 4)-shaped orbital dataframes (with
                columns "t", "x", "y", and "z"), one for each body.
            lines (List[Line2D]): Line objects representing each body's trajectory.
            title (mpl.text.Text): The title for the GIF's current frame.
            pbar (tqdm): Progress bar object to update for each iteration.

        Returns:
            List[Line2D]: Updated line objects.
        """
        for df, line in zip(dfs, lines):
            data = df[df["t"] == time_step]
            line.set_data(data.x, data.y)
            line.set_3d_properties(data.z)
        replacement = f"$${time_step}"
        pattern = r"\$\$\d+"
        new_title = re.sub(pattern, replacement, title.get_text())
        title.set_text(new_title)
        pbar.update(1)
        lines.append(title)
        return lines

    def animate_3d_orbits(self, show: bool = False) -> None:
        """Animate an orbital system, exporting the results as a GIF.

        Args:
            solved_orbiter (Orbiter): An orbital system with each body's trajectory fully mapped out over the give time span.
            show (bool): Whether to display the GIF after its generation.
        """
        r = self.solved_orbiter.r
        r_reshaped = r.transpose(
            1, 0, 2
        )  # Reshape the array from (t, N, 3) to (N, t, 3).
        dfs = [
            pd.DataFrame(body, columns=["x", "y", "z"]).rename_axis("t").reset_index()
            for body in r_reshaped
        ]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel(r"$\mathrm{x-Position\ (m)}$", labelpad=10)
        ax.set_ylabel(r"$\mathrm{y-Position\ (m)\quad\text{\ \ \ }}$", labelpad=10)
        ax.set_zlabel(r"$\mathrm{z-Position\ (m)\quad\text{\ \ \ }}$", labelpad=10)
        dt_str = f"{self.solved_orbiter.dt:.1e}".replace("e+0", "e").replace("e+", "e")
        title_txt = (
            r"$\mathrm{%s}\ $" % str(len(dfs))
            + r"$\mathrm{Orbiting\ Bodies,\ }$"
            + r"$\mathrm{Position\ vs.\ Time,\ }$"
            + r"$0Δt,\ Δt=\mathrm{(%s\ s)}$" % dt_str
        )
        title = ax.set_title(title_txt)

        lines = []
        labels = []
        handles = []
        colors = list(self.solved_orbiter.colors.values())
        for i, df in enumerate(dfs):
            label_txt = r"$m_" + str(i) + "$"
            (line,) = ax.plot(
                df.x,
                df.y,
                df.z,
                linestyle="",
                marker="o",
                label=label_txt,
                color=colors[i],
            )
            lines.append(line)
            labels.append(label_txt)
            handles.append(
                Line2D([], [], marker="", markersize=10, linestyle="-", color=colors[i])
            )
        ax.legend(
            handles=handles, labels=labels, loc="center right", bbox_to_anchor=(0, 0.5)
        )

        # Now that the figure has all lines drawn across it, animation is next.
        with tqdm(total=r.shape[0]) as pbar:
            ani = mpl.animation.FuncAnimation(
                fig,
                self.update_graph,
                fargs=(dfs, lines, title, pbar),
                frames=r.shape[0],
                interval=40,
                blit=True,
            )

            outfile = "3d_orbits.gif"
            outpath = self.solved_orbiter.outfolder / outfile
            ani.save(outpath, writer="pillow", fps=25)
            if show:
                plt.show()


@click.command()
@click.argument(
    "config_path", default="configs/4-body-test.yaml", type=click.Path(exists=True)
)
def main(config_path: str) -> None:
    """Conduct an animation test for a solved orbital system.

    Args:
        config_path (str): The path to the configuration file which specifies where to
            load a solved orbital system (among other settings).
    """
    config = load_config(config_path)
    solved_orbiter = Orbiter(config)
    animator = Animator(solved_orbiter)
    animator.animate_3d_orbits(show=True)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path[0] = str(Path(sys.path[0]).parents[0])
    from simulate_orbits import load_config

    main()
