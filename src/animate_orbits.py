from pathlib import Path
from typing import Tuple

import matplotlib as mpl
import pandas as pd
from matplotlib import pyplot as plt
from orbiter import Orbiter
from simulate_orbits import load_config
from tqdm import tqdm


def test_2_bodies(show: bool = True) -> None:
    """Test creating an animated GIF displaying the orbits of two bodies.

    Args:
        show (bool): Whether to display the GIF after its generation.
    """
    config_path = "configs/config.yaml"
    config = load_config(config_path)
    inpath = "runs/our_solar_system/2_bodies/euler_0t0_10tf_2000dt/orbital_data.pkl"
    inpath = Path(inpath)
    orbital_system = Orbiter(config)
    orbital_system.load(inpath)
    animate_3d_orbits(orbital_system)


def update_graph(
    time_step: int, df: pd.DataFrame, graph: plt.Figure, title: mpl.text.Text
) -> Tuple[mpl.text.Text, plt.Figure]:
    """Update an orbital trajectory figure's title and orbital data for one time step.

    The `update_graph` function returns objects that need to be re-drawn with each
    frame update in the animation. In the context of `FuncAnimation`, this is done to
    optimize the rendering process. Returning the objects that change allows
    `FuncAnimation` to only update those specific parts of the plot, which can
    significantly improve performance, especially for complex animations.

    Args:
        time_step (int): The integer time step.
        df (pd.DataFrame): The (t, 4)-shaped orbital dataframe (with columns "t", "x",
            "y", and "z").
        graph (plt.Figure): The figure to update.
        title (mpl.text.Text): The title for the GIF's current frame.

    Returns:
        Tuple[mpl.text.Text, plt.Figure]: The GIF's updated title and the current frame.
    """
    data = df[df["t"] == time_step]
    graph.set_data(data.x, data.y)
    graph.set_3d_properties(data.z)
    title.set_text(f"3D Orbits (t={time_step})")
    return (
        title,
        graph,
    )


def animate_3d_orbits(orbital_system: Orbiter, show: bool = False) -> None:
    """Animate an orbital system, exporting the results as a GIF.

    Args:
        orbital_system (Orbiter): A fully-solved orbital system.
        show (bool): Whether to display the GIF after its generation.
    """
    r = orbital_system.r
    r_reshaped = r.transpose(1, 0, 2)  # Reshape the array from (t, N, 3) to (N, t, 3).
    dfs = [
        pd.DataFrame(body, columns=["x", "y", "z"]).rename_axis("t").reset_index()
        for body in r_reshaped
    ]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    title = ax.set_title("3D Orbits")
    for df in tqdm(dfs):
        (graph,) = ax.plot(df.x, df.y, df.z, linestyle="", marker="o")
        ani = mpl.animation.FuncAnimation(
            fig,
            update_graph,
            fargs=(df, graph, title),
            frames=r.shape[0],
            interval=40,
            blit=False,
        )
    outfile = "3d_orbits.gif"
    outpath = orbital_system.outfolder / outfile
    ani.save(outpath, writer="pillow", fps=25)
    if show:
        plt.show()


if __name__ == "__main__":
    test_2_bodies(show=False)
