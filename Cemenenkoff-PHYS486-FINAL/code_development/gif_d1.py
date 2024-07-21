# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 07:37:25 2018

@author: Cemenenkoff
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection

class Vanishing_Line(object):
    def __init__(self, n_points, tail_length, rgb_color):
        self.n_points = int(n_points)
        self.tail_length = int(tail_length)
        self.rgb_color = rgb_color

    def set_data(self, x=None, y=None):
        if x is None or y is None:
            self.lc = LineCollection([])
        else:
            # ensure we don't start with more points than we want
            x = x[-self.n_points:]
            y = y[-self.n_points:]
            # create a list of points with shape (len(x), 1, 2)
            # array([[[  x0  ,  y0  ]],
            #        [[  x1  ,  y1  ]],
            #        ...,
            #        [[  xn  ,  yn  ]]])
            self.points = np.array([x, y]).T.reshape(-1, 1, 2)
            # group each point with the one following it (shape (len(x)-1, 2, 2)):
            # array([[[  x0  ,   y0  ],
            #         [  x1  ,   y1  ]],
            #        [[  x1  ,   y1  ],
            #         [  x2  ,   y2  ]],
            #         ...
            self.segments = np.concatenate([self.points[:-1], self.points[1:]],
                                           axis=1)
            if hasattr(self, 'alphas'):
                del self.alphas
            if hasattr(self, 'rgba_colors'):
                del self.rgba_colors
            #self.lc = LineCollection(self.segments, colors=self.get_colors())
            self.lc.set_segments(self.segments)
            self.lc.set_color(self.get_colors())

    def get_LineCollection(self):
        if not hasattr(self, 'lc'):
            self.set_data()
        return self.lc


    def add_point(self, x, y):
        if not hasattr(self, 'points'):
            self.set_data([x],[y])
        else:
            # TODO: could use a circular buffer to reduce memory operations...
            self.segments = np.concatenate((self.segments,[[self.points[-1][0],[x,y]]]))
            self.points = np.concatenate((self.points, [[[x,y]]]))
            # remove points if necessary:
            while len(self.points) > self.n_points:
                self.segments = self.segments[1:]
                self.points = self.points[1:]
            self.lc.set_segments(self.segments)
            self.lc.set_color(self.get_colors())

    def get_alphas(self):
        n = len(self.points)
        if n < self.n_points:
            rest_length = self.n_points - self.tail_length
            if n <= rest_length:
                return np.ones(n)
            else:
                tail_length = n - rest_length
                tail = np.linspace(1./tail_length, 1., tail_length)
                rest = np.ones(rest_length)
                return np.concatenate((tail, rest))
        else: # n == self.n_points
            if not hasattr(self, 'alphas'):
                tail = np.linspace(1./self.tail_length, 1., self.tail_length)
                rest = np.ones(self.n_points - self.tail_length)
                self.alphas = np.concatenate((tail, rest))
            return self.alphas

    def get_colors(self):
        n = len(self.points)
        if  n < 2:
            return [self.rgb_color+[1.] for i in range(n)]
        if n < self.n_points:
            alphas = self.get_alphas()
            rgba_colors = np.zeros((n, 4))
            # first place the rgb color in the first three columns
            rgba_colors[:,0:3] = self.rgb_color
            # and the fourth column needs to be your alphas
            rgba_colors[:, 3] = alphas
            return rgba_colors
        else:
            if hasattr(self, 'rgba_colors'):
                pass
            else:
                alphas = self.get_alphas()
                rgba_colors = np.zeros((n, 4))
                # first place the rgb color in the first three columns
                rgba_colors[:,0:3] = self.rgb_color
                # and the fourth column needs to be your alphas
                rgba_colors[:, 3] = alphas
                self.rgba_colors = rgba_colors
            return self.rgba_colors

def data_gen(t=0):
    "works like an iterable object!"
    cnt = 0
    while cnt < 1000:
        cnt += 1
        t += 0.1
        yield t, np.sin(2*np.pi*t) * np.exp(-t/100.)

def update(data):
    "Update the data, receives whatever is returned from `data_gen`"
    x, y = data
    line.add_point(x, y)
    # rescale the graph by large steps to avoid having to do it every time:
    xmin, xmax = ax.get_xlim()
    if x >= xmax:
        ax.set_xlim(xmin, 2*xmax)
        ax.figure.canvas.draw()
    return line,

if __name__ == '__main__':
    n_points = 100
    tail_length = (3/4.)*n_points
    rgb_color = [0., 0.5, 1.0]
    time_pause = 0 # miliseconds

    x=np.linspace(0, 4*np.pi, 2*n_points)
    y=np.cos(x)

    line = Vanishing_Line(n_points, tail_length, rgb_color)
    fig, ax = plt.subplots()
    ax.add_collection(line.get_LineCollection())
    ax.set_xlim(0, 4*np.pi)
    ax.set_ylim(-1.1,1.1)

    ani = animation.FuncAnimation(fig, update, data_gen, blit=False,
                                  interval=time_pause, repeat=False)

    #fig.show()

    mywriter = animation.FFMpegWriter(fps=30)
    #ani.save('ani.mp4', writer=mywriter, dpi=600)