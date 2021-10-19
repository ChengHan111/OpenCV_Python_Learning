# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 01:03:26 2021

@author: wgrim
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.optimize as opt
from scipy.interpolate import interp1d
from scipy.linalg import dft

if __name__=='__main__':
    #1) Generate the function y=0.05*x**2+x+5 over x=[-5,5], and add Student's T noise
    #with 5 degrees of freedom (df) to the function; then, use linear regression to
    #fit a line with slope and intercept to the result; plot the result in a graph
    #with the title showing the slope and intercept marked as such, x axis labeled
    #'x', y axis labeled 'y', and showing the noisy data plotted using scatter points
    #and the estimated line plotted as a line (may need to change color of line
    #or points to see it clearly, or point size, feel free to alter such parameters
    #as needed)
    x=np.arange(-5,5.01,0.01)
    y=0.05*x**2+x+5
    yn=y+stats.t.rvs(5,size=len(x))
    res=stats.linregress(x,yn)
    # this can get the linear result for slope and interception
    plt.figure()
    plt.title('Slope='+str(res.slope)+', Intercept='+str(res.intercept))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(x,yn)
    plt.plot(x,(res.slope*x)+res.intercept,'r')
    
    #2) Fit the (x,y) points (1,7),(6,6),(5,-4) to a parabola, that is, in the
    #form of a*x**2+b*x+c using curve_fit; scatter plot the points on a graph
    #and plot the resulting parabola as a line; label the x and y axes as before
    #and title the graph with the marked coefficients of the parabola; then,
    #use the root or fsolve optimization function to find the two roots of the 
    #parabola and print the result
    par=lambda x,a,b,c: a*x**2+b*x+c
    pts=np.array([[1,7],[6,6],[5,-4]])
    popt,_=opt.curve_fit(par,pts[:,0],pts[:,1]) # (par, x, y)
    prange=np.arange(0,7.01,0.01)
    plt.figure()
    plt.title('a='+str(popt[0])+', b='+str(popt[1])+', c='+str(popt[2]))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(pts[:,0],pts[:,1])
    plt.plot(prange,par(prange,popt[0],popt[1],popt[2]))
    print(opt.root(par,[0,5],args=(popt[0],popt[1],popt[2])).x)
    
    #3) Find the minimum and maximum of sin(2x)+cos(x/2) within the range x=[0,7],
    #then plot the function as a line and scatter plot the minimum and maximum
    #points; label the axes as before and title the plot with the x values of 
    #the minimum and maximum value locations
    f1=lambda x: np.sin(2*x)+np.cos(x/2)
    f2=lambda x: -f1(x)
    res1=opt.shgo(f1,bounds=[(0,7)])
    res2=opt.shgo(f2,bounds=[(0,7)])
    plt.figure()
    plt.title('Min.='+str(res1.x)+', Max.='+str(res2.x))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(prange,f1(prange))
    plt.scatter([res1.x,res2.x],[f1(res1.x),f1(res2.x)])
    
    #4) For parameters a=3, b=2, and c=1, calculate the values of a parabola
    #ONLY at integers between -5 and 5 (inclusive). Then, use interpolation to
    #determine the values of the parabola halfway between each integer (-4.5, 
    #-3.5, and so on). Finally, scatter plot the parabola at the integers and
    #then separately scatter plot the interpolated points; label them as such
    #on a legend, label the axes accordingly, and title the plot "Interpolated
    #Parabola"
    ints=np.arange(-5,6,1)
    pars=3*ints**2+2*ints+1
    parfunc=interp1d(ints,pars)
    halves=np.arange(-4.5,5.5,1)
    ph=parfunc(halves)
    plt.figure()
    plt.title('Interpolated Parabola')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(ints,pars,label='Original') # x and y with label
    plt.scatter(halves,ph,label='Interpolated')
    plt.legend()
    plt.show()
    
    #5) Use the DFT matrix from the linalg module to determine and plot the
    #Fourier Transform of a Rectangle Function
    def rectf(x):
        r=np.zeros(len(x))
        m=np.where(abs(x)<0.5)[0]
        r[m]=1
        return r
    n=300
    x=np.linspace(-5,5,n)
    s=(np.amax(x)-np.amin(x))/(n-1)
    rect=rectf(x)
    shift=lambda x: np.append(x[int(n/2):],x[:int(n/2)])
    rshift=shift(rect)
    Fshift=np.dot(dft(n),rshift) # dft is from scipy
    F=shift(Fshift)*s
    nu=x/(n*s**2)
    plt.figure()
    plt.title('Fourier Transform of a Rectangle')
    plt.xlabel('nu')
    plt.ylabel('F')
    plt.plot(nu,F)
    plt.show()