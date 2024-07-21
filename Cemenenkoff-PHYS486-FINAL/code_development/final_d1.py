# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 11:19:56 2018

@author: Cemenenkoff
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
plt.style.use('classic') #Use a serif font.
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('pdf', 'png')
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['figure.autolayout'] = False 
plt.rcParams['figure.figsize'] = 10, 6
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['font.size'] = 10
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 6
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor']='white'
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'
#This import is necessary for the isometric plot.
from mpl_toolkits.mplot3d import Axes3D
###############################################################################
N = 2 #Choose the number of bodies to explore.
###############################################################################
G = 6.67e-11 #universal gravitational constant in units of m^3/kg*s^2
end_time = 7*365.26*24*3600 # Define the end time in seconds (e.g. 7 years).
dt = 2.0*3600 #Define the length of each time step in seconds (e.g. 2 hours).
steps = int(end_time/dt) #Define the total number of time steps.
#Fill out a list of times by successively adding increments of dt.
t = dt*np.array(range(steps + 1))

#Define the masses for a number of bodies.
m0 = 1e30 #mass of body 0 in kg
m1 = 2e30 #mass of body 1 in kg, etc.
m2 = 3e30
m3 = 4e30 
m = [m0, m1, m2, m3] #Combine the masses into an ordered list.
#Set colors for the bodies.
c0 = '#E74C3C' #red
c1 = '#8E44AD' #purple
c2 = '#3498DB' #blue
c3 = '#2ECC71' #green
c = [c0, c1, c2, c3] #Put colors into a similarly ordered list.
###############################################################################
def two(m):
    #If you print out either r or v below, you'll see several "layers" of 3x3
    #matrices. Which layer you are on represents which time step you are on.
    #Within each 3x3 matrix, the row denotes which body, while the columns 1-3
    #represent x-, y-, z-positions respectively. In essence, each number in r
    #or v is associated with three indices: step #, body #, and coordinate #.
    r = np.zeros([steps+1, 2, 3]) # m
    v = np.zeros([steps+1, 2, 3]) # m/s
    
    #Next, we input initial positions. Note the first bracketed triplet of data
    #represents the x-, y-, and z-position of the first body (m0).
    r[0] = np.array([[1.0, 3.0, 2.0],
                     [6.0, -5.0, 4.0]])*1e11
    #Input initial velocities.
    v[0] = np.array([[-2.0, 0.5, 5.0],
                     [7.0, 0.5, 2.0]])*1e3
    
    def accel(r):
        a = np.zeros([2,3])
        #The acceleration of m0 has to do with the force from m1.
        #See: https://en.wikipedia.org/wiki/N-body_problem
        a[0] = G*(m[1]/LA.norm(r[0]-r[1])**3*(r[1]-r[0]))
        a[1] = G*(m[0]/LA.norm(r[1]-r[0])**3*(r[0]-r[1]))
        return a
    
    #As a first pass, we do an Euler-Cromer implementation.
    for i in range(steps):
        #The next position is the current position plus the current velocity
        #times dt.
        r[i+1] = r[i] + dt*v[i]
        #The next velocity is the current velocity plus the acceleration
        #calculated at the next position times dt.
        v[i+1] = v[i] + dt*accel(r[i+1])
    return r, v

#three() repeats the logic found in two(), but extends everything to account
#for another graviationally interacting body.
def three(m):
    r = np.zeros([steps+1, 3, 3])
    v = np.zeros([steps+1, 3, 3])
    r[0] = np.array([[1.0, 3.0, 2.0],
                     [6.0, -5.0, 4.0],
                     [7.0, 8.0, -7.0]])*1e11
    v[0] = np.array([[-2.0, 0.5, 5.0],
                     [7.0, 0.5, 2.0],
                     [-4.0, -0.5, -3.0]])*1e3
    def accel(r):
        a = np.zeros([3,3])
        #The acceleration of m0 has to do with forces from m1 and m2.
        a[0] = G*(m[1]/LA.norm(r[0]-r[1])**3*(r[1]-r[0])
                + m[2]/LA.norm(r[0]-r[2])**3*(r[2]-r[0]))
        
        a[1] = G*(m[0]/LA.norm(r[1]-r[0])**3*(r[0]-r[1])
                + m[2]/LA.norm(r[1]-r[2])**3*(r[2]-r[1]))
        
        a[2] = G*(m[0]/LA.norm(r[2]-r[0])**3*(r[0]-r[2])
                + m[1]/LA.norm(r[2]-r[1])**3*(r[1]-r[2]))
        return a
    for i in range(steps):
        r[i+1] = r[i] + dt*v[i]
        v[i+1] = v[i] + dt*accel(r[i+1])
    return r, v

#four() extends the problem to four gravitationally interacting bodies.
def four(m):
    r = np.zeros([steps+1, 4, 3])
    v = np.zeros([steps+1, 4, 3])
    #These initial conditions make a cool pattern.
    r[0] = np.array([[0.0, 0.0, 0.0],
                     [1.0, 3.0, 2.0],
                     [6.0, -5.0, 4.0],
                     [7.0, 8.0, -7.0]])*1e11
    v[0] = np.array([[0.01, 0.0, 0.4],
                     [0.0, 0.1, 0.0],
                     [3.0, 0.0, 1.0],
                     [1.0, 1.0, 1.0]])*1e2
    def accel(r):
        a = np.zeros([4,3])
        a[0] = G*(m[1]/LA.norm(r[0]-r[1])**3*(r[1]-r[0])
                + m[2]/LA.norm(r[0]-r[2])**3*(r[2]-r[0])
                + m[3]/LA.norm(r[0]-r[3])**3*(r[3]-r[0]))
        
        a[1] = G*(m[0]/LA.norm(r[1]-r[0])**3*(r[0]-r[1])
                + m[2]/LA.norm(r[1]-r[2])**3*(r[2]-r[1])
                + m[3]/LA.norm(r[1]-r[3])**3*(r[3]-r[1]))
        
        a[2] = G*(m[0]/LA.norm(r[2]-r[0])**3*(r[0]-r[2])
                + m[1]/LA.norm(r[2]-r[1])**3*(r[1]-r[2])
                + m[3]/LA.norm(r[2]-r[3])**3*(r[3]-r[2]))
        
        a[3] = G*(m[0]/LA.norm(r[3]-r[0])**3*(r[0]-r[3])
                + m[1]/LA.norm(r[3]-r[1])**3*(r[1]-r[3])
                + m[2]/LA.norm(r[3]-r[2])**3*(r[2]-r[3]))
        return a
    for i in range(steps):
        r[i+1] = r[i] + dt*v[i]
        v[i+1] = v[i] + dt*accel(r[i+1])
    return r, v
###############################################################################
if N == 2:
    r, v = two(m)
    Nstr = '$\mathrm{Two\ }$'
elif N == 3:
    r, v = three(m)
    Nstr = '$\mathrm{Three\ }$'
elif N == 4:
    r, v = four(m)
    Nstr = '$\mathrm{Four\ }$'
###############################################################################
fig1 = plt.figure(1, facecolor='white')
ax1 = fig1.add_subplot(1,1,1, projection='3d')
plt.title(r'%s'%Nstr+r'$\mathrm{Orbiting\ Bodies}$', y=1.05)
ax1.set_xlabel(r'$\mathrm{x-position}\ \mathrm{(m)}$', labelpad=10)
ax1.set_ylabel(r'$\mathrm{y-position}\ \mathrm{(m)}$', labelpad=10)
ax1.set_zlabel(r'$\mathrm{z-position}\ \mathrm{(m)}$', labelpad=10)
#For all times, plot (x,y,z) tuples for m0.
ax1.plot(r[:, 0, 0], r[:, 0, 1], r[:, 0, 2], color=c[0], label=r'$m_0$')
#For all times, plot (x,y,z) tuples for m1, etc.
ax1.plot(r[:, 1, 0], r[:, 1, 1], r[:, 1, 2], color=c[1], label=r'$m_1$')
if N > 2:
    ax1.plot(r[:, 2, 0], r[:, 2, 1], r[:, 2, 2], color=c[2], label=r'$m_2$')
if N > 3:
    ax1.plot(r[:, 3, 0], r[:, 3, 1], r[:, 3, 2], color=c[3], label=r'$m_3$')
ax1.axis('equal')
plt.legend(loc='upper left')
###############################################################################
fig2 = plt.figure(2, facecolor='white')
ax2 = fig2.add_subplot(111)
plt.title(r'%s'%Nstr
          +r'$\mathrm{Orbiting\ Bodies\ Viewed \ Along\ the\ y-Axis}$', y=1.05)
ax2.set_xlabel(r'$\mathrm{x-position}\ \mathrm{(m)}$')
ax2.set_ylabel(r'$\mathrm{z-position}\ \mathrm{(m)}$')
#For all times, plot (x,z) tuples for m0.
ax2.plot(r[:, 0, 0], r[:, 0, 2], color=c[0], label=r'$m_0$')
#For all times, plot (x,z) tuples for m1, etc.
ax2.plot(r[:, 1, 0], r[:, 1, 2], color=c[1], label=r'$m_1$')
if N > 2:
    ax2.plot(r[:, 2, 0], r[:, 2, 2], color=c[2], label=r'$m_2$')
if N > 3:
    ax2.plot(r[:, 3, 0], r[:, 3, 2], color=c[3], label=r'$m_3$')
ax2.axis('equal')
ax2.legend(loc='lower right')