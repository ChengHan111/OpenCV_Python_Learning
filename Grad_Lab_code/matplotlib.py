# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 21:14:40 2021

@author: wgrim
"""

import numpy as np
import matplotlib.pyplot as plt

#Can use this for MATLAB similarities, but I recommend matplotlib
from pylab import *

#Here's one way to create a basic plot
x=np.arange(0,10,0.01)
y=x**2
plt.figure() #Necessary or everything stacks into one figure
plt.title('My Plot')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x,y,'r',label='function 1') #'r','b' examples of color keywords
plt.plot(x,x,'b',label='function 2') #label shown in legend
plt.legend()
plt.show() #Necessary for Jupyter Notebook

#This is an object-based way to make a figure with subplots
#(or nest other graphs in by altering the axis coordinates)
fig=plt.figure()
ax=fig.add_axes([0,0,1,0.5]) #[left, bottom, width, height], all (0-1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('My Object Plot')
ax.plot(x,x**2)
ax2=fig.add_axes([0,0.5,1,0.5])
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('My Object Plot')
ax2.plot(x,x)
fig.show()

#Alternatively, you can use this for subplots
fig,ax=plt.subplots(2,1,sharex=True) #Rows, columns
ax[0].set_ylabel('y')
ax[0].set_title('Top Plot')
ax[0].plot(x,x**2)
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')
ax[1].set_title('Bottom Plot')
ax[1].plot(x,x)
fig.show()

#Save figures to the current directory
plt.savefig('stacked_plot.png')

#Use $ symbols for LateX font!
fig,ax=plt.subplots(2,1,sharex=True)
ax[0].set_ylabel(r'$\beta$')
ax[0].set_title(r'$Top$ $Plot$')
ax[0].plot(x,x**2)
ax[1].set_xlabel(r'$\alpha$')
ax[1].set_ylabel(r'$\beta$')
ax[1].set_title(r'$Bottom$ $Plot$')
ax[1].plot(x,x)
fig.show()

#This allows the use of other fonts, like STIX, via altering the
#parameter dictionary
matplotlib.rcParams.update({'font.size':20,'font.family':'STIXGeneral',\
                           'mathtext.fontset':'stix'})
fig,ax=plt.subplots(2,1,sharex=True)
ax[0].set_ylabel(r'$\beta$')
ax[0].set_title(r'$Top$ $Plot$')
ax[0].plot(x,x**2)
ax[1].set_xlabel(r'$\alpha$')
ax[1].set_ylabel(r'$\beta$')
ax[1].set_title(r'$Bottom$ $Plot$')
ax[1].plot(x,x)
fig.show()

#Change the font back
matplotlib.rcParams.update({'font.size':12,'font.family':'sans',\
                           'text.usetex':False})
						   
#Use alpha (0-1) to alter line/point transparency, hexcodes in
#hexadecimal format for specific colors
fig,ax=plt.subplots(2,1,sharex=True)
ax[0].set_ylabel(r'$\beta$')
ax[0].set_title(r'$Top$ $Plot$')
ax[0].plot(x,x**2,alpha=0.25,color='#1155dd')
ax[1].set_xlabel(r'$\alpha$')
ax[1].set_ylabel(r'$\beta$')
ax[1].set_title(r'$Bottom$ $Plot$')
ax[1].plot(x,x,alpha=1,color='g')
fig.show()

#Create scatter plots, alter the line style (ls) and width (lw), and
#marker shape (marker) and size (ms, s)!
fig,ax=plt.subplots(2,1,sharex=True)
ax[0].set_ylabel(r'$\beta$')
ax[0].set_title(r'$Top$ $Plot$')
ax[0].plot(x,x**2,lw=0.75,ls='-.',marker='o',ms=0.75)
ax[0].scatter(x,x**2,marker='o',s=0.75,alpha=0.25)
ax[1].set_xlabel(r'$\alpha$')
ax[1].set_ylabel(r'$\beta^2$')
ax[1].set_title(r'$Bottom$ $Plot$')
ax[1].plot(x,x,lw=1.5,ls='dotted',marker='^',ms=0.75)
ax[1].scatter(x,x,marker='^',s=0.75,alpha=0.25)
fig.show()

#Force x and y limits as needed for graphs
fig,ax=plt.subplots(2,1,sharex=True)
ax[0].set_ylabel(r'$\beta$')
ax[0].set_title(r'$Top$ $Plot$')
ax[0].plot(x,x**2)
ax[1].set_xlabel(r'$\alpha$')
ax[1].set_ylabel(r'$\beta$')
ax[1].set_title(r'$Bottom$ $Plot$')
ax[1].plot(x,x)
ax[0].set_xlim(2,5)
ax[1].set_xlim(2,5)
ax[0].set_ylim(2,5)
ax[1].set_ylim(2,5)
#plt.xlim(2,5)
#plt.ylim(2,5)
fig.show()

#Set x (and y) ticks to specific values and labels
fig,ax=plt.subplots(2,1,sharex=True)
ax[0].set_ylabel(r'$\beta$')
ax[0].set_title(r'$Top$ $Plot$')
ax[0].plot(x,x**2)
ax[1].set_xlabel(r'$\alpha$')
ax[1].set_ylabel(r'$\beta$')
ax[1].set_title(r'$Bottom$ $Plot$')
ax[1].plot(x,x)
plt.xticks([1,2,3,4,5],['A','B','C','D','E'])

#Bar graph
plt.figure()
plt.xticks([1,2,3,4,5],['A','B','C','D','E'])
plt.bar([1,2,3,4,5],[5,4,3,2,1])
plt.title('Bar Plot')
plt.xlabel('x')
plt.ylabel('y')

#Step graph
plt.figure()
plt.xticks([1,2,3,4,5],['A','B','C','D','E'])
plt.step([1,2,3,4,5],[5,4,3,2,1])
plt.title('Step Plot')
plt.xlabel('x')
plt.ylabel('y')

#Fill-between graph
plt.figure()
plt.title('Fill')
plt.xlabel('x')
plt.ylabel('y')
plt.fill_between(x,x**2,x)
plt.show()

#Error Bar Plot (can use yerr keyword for y-error bars)
plt.figure()
plt.title('Errorbar')
plt.xlabel('x')
plt.ylabel('y')
plt.errorbar(x,y,xerr=np.ones(len(x))*0.1)
plt.show()

#Histograms, regular and cumulative
arr=np.random.standard_normal(10000)
plt.figure()
plt.hist(arr,bins=200)
plt.figure()
plt.hist(arr,bins=200,cumulative=True)

#Log plots (xscale, semilogx, loglog)
plt.figure()
plt.title('Log')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x,y)
plt.xscale('log')
plt.yscale('log')
#plt.loglog(x,y)
#plt.semilogx(x,y)
#plt.semilogy(x,y)
plt.show()

#Text annotations on graph
plt.figure()
plt.title('Log')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x,y)
plt.text(4,40,r'$y=x^2$') #based on graph coordinates
plt.show()

#Make and show 2-D functions or images like so:
cos2d=lambda x,y: np.cos(x)+np.cos(y)
t=np.arange(-5,5,0.01)
T1,T2=np.meshgrid(t,t) #x coords., y coords.
plt.imshow(cos2d(T1,T2),cmap='gray') #cmap for different color maps
plt.colorbar() #Colorbar to show color for all values

#Contours on a 2D plot
plt.contour(cos2d(T1,T2),cmap='gray')

#3D plots like so
from mpl_toolkits.mplot3d.axes3d import Axes3D
fig3d=plt.figure()
ax=fig3d.add_subplot(1,2,1,projection='3d') #rows, columns, index, make 3D
ax.plot_surface(T1,T2,cos2d(T1,T2)) #X,Y,Z
ax.view_init(90,0) #Change viewing angle of 3D plot

#Pie charts like so
plt.pie(np.linspace(0,5,10))

#Polar coordinate maps like so
plt.polar(x,y)