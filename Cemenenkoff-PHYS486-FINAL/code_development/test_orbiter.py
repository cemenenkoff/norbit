# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 11:19:56 2018

@author: Cemenenkoff
"""

from udacityplots import *
import numpy
import matplotlib
#from mpl_toolkits.mplot3d import Axes3D

m0 = 1e30 # kg
m1 = 2e30 # kg
m2 = 3e30 # kg
m = [m0, m1, m2]
G = 6.67e-11 # m^3/kg*s^2

end_time = 10*365.26*24*3600 # ten years in seconds
dt = 2.0*3600 # 2 hours in seconds (length of time step)
steps = int(end_time/dt) #number of time steps
#times = dt * numpy.array(range(steps + 1)) #fill out a list of times

def three_body_problem():
    #If you print out either r or v below, you'll see several "layers" of 3x3
    #matrices. Which layer you are on represents which time step you are on.
    #Within each 3x3 matrix, the row denotes which body, while the columns 1-3
    #represent x-, y-, z-positions respectively. In essence, each number in r
    #or v is associated with three indices: step #, body #, and coordinate #.
    r = numpy.zeros([steps + 1, 3, 3]) # m
    v = numpy.zeros([steps + 1, 3, 3]) # m/s
    
    #Next, we input initial positions. Note the first bracketed triplet of data
    #represents the x-, y-, and z-position of the first body (m0).
    r[0] = numpy.array([[1.0, 3.0, 2.0],
                       [6.0, -5.0, 4.0],
                       [7.0, 8.0, -7.0]])*1e11
    #Input initial velocities.
    v[0] = numpy.array([[-2.0, 0.5, 5.0],
                       [7.0, 0.5, 2.0],
                       [-4.0, -0.5, -3.0]])*1e3 #initial velocities
    
    from numpy import linalg as LA
    
    def acceleration(r):
        a = numpy.zeros([3,3])
        #a[0] = G*m[1]/LA.norm(r[0]-r[1])**3*(r[1]-r[0])+G*m[2]/LA.norm(r[0]-r[2])**3*(r[2]-r[0])
        a[0] = G*(m[1]/LA.norm(r[0]-r[1])**3*(r[1]-r[0]) + m[2]/LA.norm(r[0]-r[2])**3*(r[2]-r[0]))
        
        
        a[1] = G*m[0]/LA.norm(r[1]-r[0])**3*(r[0]-r[1])+G*m[2]/LA.norm(r[1]-r[2])**3*(r[2]-r[1])
        a[2] = G*m[0]/LA.norm(r[2]-r[0])**3*(r[0]-r[2])+G*m[1]/LA.norm(r[2]-r[1])**3*(r[1]-r[2])
        return a
    
    for i in range(steps):
        r[i+1] = r[i] + dt*v[i]
        v[i+1] = v[i] + dt*acceleration(r[i+1])
    return r, v

r, v = three_body_problem()

@show_plot
def plot_stars():
    axes = matplotlib.pyplot.gca()
    axes.set_xlabel('x in m')
    axes.set_ylabel('z in m')
    axes.plot(r[:, 0, 0], r[:, 0, 2])
    axes.plot(r[:, 1, 0], r[:, 1, 2])
    axes.plot(r[:, 2, 0], r[:, 2, 2])
    matplotlib.pyplot.axis('equal')
    
plot_stars()