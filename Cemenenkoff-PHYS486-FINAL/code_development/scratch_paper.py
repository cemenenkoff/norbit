# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 05:10:08 2018

@author: Cemenenkoff
"""

"""
N_ = 5
words = list(range(2,N_+1))
words = [str(x) for x in words]
print(words)#"""

n=8
print(n%2)
#for i starting at 1, going up to n, skipping by 2 each time (odd #s)...
for i in range(1, n, 2):
    print('hi',i)
#for i starting at 2, going up to n-1, skipping by 2 each time (even #s)...
for i in range(2, n-1, 2):
    print('bi',i)

"""
import numpy as np
from numpy import linalg as LA
G=1
N=3
steps=9
m=np.array([1,2,3])
r = np.ones([N, 3])
print(r, np.shape(r))
print('\n')

a=np.zeros([N,3])
U=np.zeros([N,1])
#print(a, np.shape(a))
#print('\n')
for i in range(N):
    #print(a[i], np.shape(a[i]), np.shape(r[i]))
    print('~\n')
    j = list(range(i))+list(range(i+1,N)) #generate the popped list
    for k in range(N-1): #for each number in the popped list
        #pass
        #a[i,:]=G*(m[j[k]]/LA.norm(r[i]-r[j[k]])**3*(r[j[k]]-r[i]))+a[i]
        #a[i]=G*(m[j[k]]/LA.norm(r[i]-r[j[k]])**3*(r[j[k]]-r[i]))+a[i]
        U[i]=-G*m[j[k]]*m[i]/LA.norm(r[i]-r[j[k]])+U[i]
        print(np.shape(U[i]))#, np.shape(r[i]), np.shape(r[j[k]]))
        #terms[k]=(j[k], i, j[k], j[k], i)
        #terms[k]=(m[j[k]], i, j[k], j[k], i)
        #terms[k]=(m[j[k]]+r[i]+r[j[k]]+r[j[k]]+r[i])
        #terms[k] = m[j[k]]/LA.norm(r[i]-r[j[k]])**3*(r[j[k]]-r[i])
        
        #print(terms)
        #print('\n')
        #a[i] = G*np.sum(terms)
    #a[i] = G*np.sum(terms)
    #print(i, j, terms, a[i])
    #print('\n')
    
print(U)#"""


#a=np.zeros([N,3])
#for i in range(N):
#    terms = [None]*(N-1) # no. terms in the acceleration calculation
#    j = list(range(i))+list(range(i+1,N))
#    for k in range(len(j)):
#        terms[k] = m[j[k]]/LA.norm(r[i]-r[j[k]])**3*(r[j[k]]-r[i])
#    a[i] = G*np.sum(terms)








"""
import numpy as np
G=1
N=3
m=[2,3,5,7,11]
a=np.zeros([N,3])
for i in range(N):
    terms = [None]*(N-1)
    #print(terms)
    k = list(range(i))+list(range(i+1,N))
    #print(k)
    for j in range(len(k)):
        #terms[j] = m[j]/LA.norm(r[i]-r[j])**3*(r[j]-r[i])
        #terms[j]=(k[j], i, k[j], k[j], i)
        #terms[j]=(m[k[j]], i, k[j], k[j], i)
        terms[j]=(m[k[j]]+i+k[j]+k[j]+i)
    a[i] = G*np.sum(terms)
    #a[i] = np.sum(terms)
    #print(i, k, terms)
    print(i, k, terms, a[i])
#print(a)
    
#"""

"""
test = [1, 1, 1, 1]
for j in range(len(test)):
    print(test[j])"""

#N=5
#import numpy as np
#terms = np.zeros(N-1)
#terms[0]=1
#terms[1]=2
#terms[2]=3
#terms[3]=4
#print(terms)

#x = 1
#print(-x)

    
    
    
    
"""
import numpy as np
steps=7
N=3

m0 = 1
m1 = 2
m2 = 3

m = [m0, m1, m2]

v = 2*np.ones((steps+1,N,3))
#v = np.array([[[1., 2., 3.],
#               [1., 2., 3.],
#               [1., 2., 3.]]])
print('This is v by itself')
print(v)
print('\n')
print('FLAG')
print(v[4:])
print('\n')
v2 = v**2
print('This is v**2')
print(v2)
print('\n')

print('On the first time step, for m0, sum v2 in the x-,y-,and z-directions.')
s = sum(v2[0,0,:])
print(s)
print('\n')

KE = np.zeros((steps+1,N,1))
p = np.zeros((steps+1,N,3))

for i in range(steps+1):
    for j in range(N):
        KE[i,j,0] = (m[j]/2)*sum(v2[i,j,:]) #(1/2)m(vx^2+vy^2+vz^2)
        p[i,j,0] = m[j]*v[i,j,0] #px
        p[i,j,1] = m[j]*v[i,j,1] #py
        p[i,j,2] = m[j]*v[i,j,2] #
print('momentum data')
print(p)
print('\n')

print('KE data')
print(KE)#"""


"""
r0 = np.array([[  1.0,  3.0,  2.0 ],
               [  6.0, -5.0,  4.0 ],
               [  7.0,  8.0, -7.0 ],
               [  8.0,  9.0, -6.0 ],
               [  8.8,  9.8, -6.8 ]])*1e11
v0 = np.array([[  0.0,  0.0,  0.0 ],
               [  7.0,  0.5,  2.0 ],
               [ -4.0, -0.5, -3.0 ],
               [  7.0,  0.5,  2.0 ],
               [  7.8,  1.3,  2.8 ]])*1e3
    
IC = np.array((r0,v0))
print('here is IC[0,0:N] aka r[0]')
print(IC[0,0:N])
print('\n')

print('here is IC[1,0:N] aka v[0]')
print(IC[1,0:N])
print('\n')


r = np.zeros([t, N, 3])
v = np.zeros([t, N, 3])

r[0] = IC[0,0:N]
v[0] = IC[1,0:N]

print('here is r[0]')
print(r[0])
print('here is v[0]')
print(v[0])#"""









































            