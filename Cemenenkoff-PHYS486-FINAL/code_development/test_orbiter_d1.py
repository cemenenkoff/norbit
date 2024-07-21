# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 11:19:56 2018

@author: Cemenenkoff
"""

import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt
plt.style.use('classic') #Use a serif font.
from mpl_toolkits.mplot3d import Axes3D

N = 3 #Choose the number of bodies to explore.

G = 6.67e-11 #universal gravitational constant in units of m^3/kg*s^2
end_time = 7*365.26*24*3600 # Define the end time in seconds (e.g. 10 years).
dt = 2.0*3600 #Define the length of each time step in seconds (e.g. 2 hours).
steps = int(end_time/dt) #Define the total number of time steps.
#Fill out a list of times by successively adding increments of dt.
t = dt*np.array(range(steps + 1))

#Define the masses for a number of bodies.
m0 = 1e30 #mass of body 0 in kg
m1 = 2e30 #mass of body 1 in kg, etc.
m2 = 3e30
m3 = 4e30 
m = [m0, m1, m2, m3] #Combine the masses into a list.

#Set colors for the bodies.
c0 = '#E74C3C' #red
c1 = '#8E44AD' #purple
c2 = '#3498DB' #blue
c3 = '#2ECC71' #green
c = [c0, c1, c2, c3] #Put colors into a similarly ordered list.

def three(m):
    #If you print out either r or v below, you'll see several "layers" of 3x3
    #matrices. Which layer you are on represents which time step you are on.
    #Within each 3x3 matrix, the row denotes which body, while the columns 1-3
    #represent x-, y-, z-positions respectively. In essence, each number in r
    #or v is associated with three indices: step #, body #, and coordinate #.
    r = np.zeros([steps+1, 3, 3]) # m
    v = np.zeros([steps+1, 3, 3]) # m/s
    
    #Next, we input initial positions. Note the first bracketed triplet of data
    #represents the x-, y-, and z-position of the first body (m0).
    r[0] = np.array([[1.0, 3.0, 2.0],
                     [6.0, -5.0, 4.0],
                     [7.0, 8.0, -7.0]])*1e11
    #Input initial velocities.
    v[0] = np.array([[-2.0, 0.5, 5.0],
                     [7.0, 0.5, 2.0],
                     [-4.0, -0.5, -3.0]])*1e3
    
    
    def accel(r):
        a = np.zeros([3,3])
        #The acceleration of m0 has to do with forces from m1 and m2.
        #See: https://en.wikipedia.org/wiki/N-body_problem
        a[0] = G*(m[1]/LA.norm(r[0]-r[1])**3*(r[1]-r[0])
                + m[2]/LA.norm(r[0]-r[2])**3*(r[2]-r[0]))
        
        a[1] = G*(m[0]/LA.norm(r[1]-r[0])**3*(r[0]-r[1])
                + m[2]/LA.norm(r[1]-r[2])**3*(r[2]-r[1]))
        
        a[2] = G*(m[0]/LA.norm(r[2]-r[0])**3*(r[0]-r[2])
                + m[1]/LA.norm(r[2]-r[1])**3*(r[1]-r[2]))
        return a
    
    #As a first pass, we do an Euler implementation.
    for i in range(steps):
        #The next position is the current position plus the current velocity
        #times dt.
        r[i+1] = r[i] + dt*v[i]
        #The next velocity is the current velocity plus the acceleration
        #calculated at the next position times dt.
        v[i+1] = v[i] + dt*accel(r[i+1])
    return r, v

#four() repeats the logic found in three(), but extends everything to account
#for another graviationally interacting body.
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

if N == 3:
    r, v = three(m)
elif N == 4:
    r, v = four(m)

fig1 = plt.figure(1, facecolor='white')
ax1 = fig1.add_subplot(1,1,1, projection='3d')
plt.title(r'$\mathrm{Three\ Orbiting\ Bodies}$', y=1.05)
ax1.set_xlabel(r'$\mathrm{x-position}\ \mathrm{(m)}$')
ax1.set_ylabel(r'$\mathrm{y-position}\ \mathrm{(m)}$')
ax1.set_zlabel(r'$\mathrm{z-position}\ \mathrm{(m)}$')
#For all times, plot (x,y,z) tuples for m0.
ax1.plot(r[:, 0, 0], r[:, 0, 1], r[:, 0, 2], color=c[0])
#For all times, plot (x,y,z) tuples for m1, etc.
ax1.plot(r[:, 1, 0], r[:, 1, 1], r[:, 1, 2], color=c[1])
ax1.plot(r[:, 2, 0], r[:, 2, 1], r[:, 2, 2], color=c[2])
if N == 4:
    ax1.plot(r[:, 3, 0], r[:, 3, 1], r[:, 3, 2], color=c[3])
ax1.axis('equal')

fig2 = plt.figure(2, facecolor='white')
ax2 = fig2.add_subplot(111)
plt.title(r'$\mathrm{Three\ Orbiting\ Bodies\ Viewed\ Along\ the\ y-Axis}$',
          y=1.05)
ax2.set_xlabel(r'$\mathrm{x-position}\ \mathrm{(m)}$')
ax2.set_ylabel(r'$\mathrm{z-position}\ \mathrm{(m)}$')
#For all times, plot (x,z) tuples for m0.
ax2.plot(r[:, 0, 0], r[:, 0, 2], color=c[0])
#For all times, plot (x,z) tuples for m1, etc.
ax2.plot(r[:, 1, 0], r[:, 1, 2], color=c[1])
ax2.plot(r[:, 2, 0], r[:, 2, 2], color=c[2])
if N == 4:
    ax2.plot(r[:, 3, 0], r[:, 3, 2], color=c[3])
ax2.axis('equal')

#I THINK this is going to be obsolete, but it is saved here just in case.
"""
r, v = four(m)

fig3 = plt.figure(3, facecolor='white')
ax3 = fig3.add_subplot(1,1,1, projection='3d')
plt.title(r'$\mathrm{Four\ Orbiting\ Bodies}$', y=1.05)
ax3.set_xlabel(r'$\mathrm{x-position}\ \mathrm{(m)}$')
ax3.set_ylabel(r'$\mathrm{y-position}\ \mathrm{(m)}$')
ax3.set_zlabel(r'$\mathrm{z-position}\ \mathrm{(m)}$')
#For all times, plot (x,y,z) tuples for m0.
ax3.plot(r[:, 0, 0], r[:, 0, 1], r[:, 0, 2], color=c[0])
#For all times, plot (x,y,z) tuples for m1, etc.
ax3.plot(r[:, 1, 0], r[:, 1, 1], r[:, 1, 2], color=c[1])
ax3.plot(r[:, 2, 0], r[:, 2, 1], r[:, 2, 2], color=c[2])
ax3.plot(r[:, 3, 0], r[:, 3, 1], r[:, 3, 2], color=c[3])
ax3.axis('equal')

fig4 = plt.figure(4, facecolor='white')
ax4 = fig4.add_subplot(111)
plt.title(r'$\mathrm{Four\ Orbiting\ Bodies\ Viewed\ Along\ the\ y-Axis}$',
          y=1.05)
ax4.set_xlabel(r'$\mathrm{x-position}\ \mathrm{(m)}$')
ax4.set_ylabel(r'$\mathrm{z-position}\ \mathrm{(m)}$')
#For all times, plot (x,z) tuples for m0.
ax4.plot(r[:, 0, 0], r[:, 0, 2], color=c[0])
#For all times, plot (x,z) tuples for m1, etc.
ax4.plot(r[:, 1, 0], r[:, 1, 2], color=c[1])
ax4.plot(r[:, 2, 0], r[:, 2, 2], color=c[2])
ax4.plot(r[:, 3, 0], r[:, 3, 2], color=c[3])
ax4.axis('equal')#"""