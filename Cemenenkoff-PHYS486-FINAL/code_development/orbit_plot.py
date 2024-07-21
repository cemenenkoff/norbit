# -*- coding: utf-8 -*-
"""
Created on Sun May 20 21:39:09 2018

@author: Cemenenkoff
"""

import numpy as np

v = np.zeros([7, 3, 3])
x = np.zeros([4,3])

v[0][2][0]=1
v[0][2][1]=2
v[0][2][2]=3
v[0][1][0]=4
v[0][1][1]=5
v[0][1][-1]=6
v[0][0][0]=7
v[0][0][1]=8
v[0][0][2]=9

v[1]=np.pi


v[0] = np.array([[1., 3., 2.], [6., -5., 4.], [7., 8., -7.]]) * 1e11

#print(v, len(v))
print(x)

r = np.zeros([7, 3, 3]) # m
v = np.zeros([7, 3, 3]) # m / s
r[0] = np.array([[1.0, 3.0, 2.0], [6.0, -5.0, 4.0], [7.0, 8.0, -7.0]])*1e11 #initial positions
v[0] = np.array([[-2.0, 0.5, 5.0], [7.0, 0.5, 2.0], [-4.0, -0.5, -3.0]])*1e3 #initial velocities

print(r)
print('hi')
print(v)

t = 0.1*np.array(range(10 + 1)) #fill out a list of times

print(t)