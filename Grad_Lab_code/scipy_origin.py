# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 15:03:37 2021

@author: wgrim
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.optimize as opt
from scipy.interpolate import interp1d
import scipy.signal as sig
import scipy.linalg as lng

#use linregress to perform linear regression and fit data to a line
x=np.linspace(-5,5,100)
y=2*np.exp(0.25*x)+5
yn=y+stats.norm.rvs(size=len(x)) #random variable sample
res=stats.linregress(x,yn)
moment=stats.norm.moment(1) #extra arguments may be needed based on distribution;
#1=mean, 2=variance, 3=skewness, 4=kurtosis (tail of distribution)
plt.figure()
plt.scatter(x,yn) 
plt.plot(x,res.slope*x+res.intercept)

#Use optimization routines to solve for the exact roots of a function, like root
#or fsolve, with initial guesses
par=lambda x,a,b,c: a*x**2+b*x+c
x1=np.linspace(-6,6,1000)
f=par(x1,1,6,4)
p_res=opt.root(par,[-6,-2],args=(1,6,4))
solved=opt.fsolve(par,[-6,-2],args=(1,6,4))

#Use global optimization routines like differential_evolution or shgo to find
#global minima within independent variable bounds (no need for initial guess!)
g_res=opt.differential_evolution(par,[(-6,6)],args=(1,6,4)) #or shgo

#Make the function negative to find a global maximum instead!
npar=lambda x,a,b,c: -par(x,a,b,c)
max_res=opt.differential_evolution(npar,[(-6,6)],args=(-1,6,4))

#Use curve_fit to fit defined functions to given data points by solving for 
#coefficients!
poly3=lambda x,a,b,c,d: a*x**3+b*x**2+c*x+d
arr=np.random.randint(0,10,(4,2)) # 随机数来fit的
# print('arr:',arr)
nvars,_=opt.curve_fit(poly3,arr[:,0],arr[:,1])
x2=np.linspace(-1,8,1000)
plt.figure()
plt.scatter(arr[:,0],arr[:,1]) 
plt.plot(x2,poly3(x2,nvars[0],nvars[1],nvars[2],nvars[3]))

#This function creates polynomials of degree based on the length of the input vector
#i.e. x**2+... if 3
func=np.poly1d([4,3,2])

#Interpolate using scipy.interpolate.interp1d
x3=np.linspace(-5,5,10)
y=x3**2
parab_interp=interp1d(x3,y) #fill_value='extrapolate'
plt.figure()
plt.scatter(x3,y) 
plt.plot(x,parab_interp(x))

#Use scipy.signal to perform convolution or correlation

def rect(x):
    r=np.empty(len(x))
    for i in np.arange(0,len(x),1):
        if abs(x[i])>0.5:
            r[i]=0
        else:
            r[i]=1
    return r

def tri(x):
    t=np.empty(len(x))
    for i in np.arange(0,len(x),1):
        if abs(x[i])>1:
            t[i]=0
        elif x[i]<0:
            t[i]=x[i]+1
        elif x[i]>=0:
            t[i]=-x[i]+1
    return t

xt=np.linspace(-1,1,100)
conv1=sig.convolve(rect(x),tri(xt),mode='same') #or fftconvolve for large signals or images
conv2=sig.correlate(rect(x),tri(xt),mode='same')

plt.figure()
plt.plot(x,conv1)

#Use np (or scipy) linalg module for linear algebra calculations

#Norm
norm=lng.norm(rect(x),3) #L3 norm; use 2 for L2, 1 fr L1, etc.

#Inverse and pseudoinverse
arr_rand=np.random.randint(0,10,(4,4))
arinv=lng.inv(arr_rand)
arr_rander=np.random.randint(0,10,(4,6))
arpinv=lng.pinv(arr_rander)

#Circulant Matrix
circ=lng.circulant(np.random.randint(0,10,8))

#Calculate the Fourier Transform of a function
s=0.01 #spacing
x4=np.arange(-5,5,s)
tri4=tri(x4)
D=lng.dft(len(x4)) #Discrete Fourier Transform matrix
shift=lambda x: np.append(x[int(len(x)/2):],x[:int(len(x)/2)]) #Switch halves of vector
tri_shift=shift(tri4) #shift vector before FT
Ftri_s=np.dot(D,tri_shift) #Fourier Transform by multiplying DFT mat. with vector
Ft=shift(Ftri_s)*s #shift resulting vector, then scale by x spacing
nu=x4/(len(x4)*s**2) #frequency=x/(len(x)*spacing**2)
plt.figure()
plt.plot(nu,Ft) 
plt.plot(nu,np.sinc(nu)**2) #proof of function

#from scipy.io import loadmat
#loadmat loads in MATLAB files