import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation
import pandas as pd



def animate_3d_orbits(r, t):
    df = pd.DataFrame({"time": t ,"x" : r[:,0], "y" : r[:,1], "z" : r[:,2]})

    def update_graph(num):
        data=df[df['time']==num]
        graph.set_data (data.x, data.y)
        graph.set_3d_properties(data.z)
        title.set_text('3D Test, time={}'.format(num))
        return title, graph,

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    title = ax.set_title('3D Test')
    data=df[df['time']==0]
    graph, = ax.plot(data.x, data.y, data.z, linestyle="", marker="o")
    ani = matplotlib.animation.FuncAnimation(fig, update_graph, 19,
                                interval=40, blit=True)

    plt.show()