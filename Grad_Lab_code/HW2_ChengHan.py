# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 23:58:51 2021

@author: wgrim
"""
import numpy as np
import matplotlib.pyplot as plt

#1) Plot y=cos(2x) from x=-5 to x=5 (in radians; numpy cosine function
#is in radians by default); label axes with 'x' and 'y' and title the graph
#'Cosine Plot'
x=np.arange(-5,5,0.01)
y=np.cos(2*x)
plt.figure()
plt.title('Cosine Plot')
plt.plot(x,y)
plt.xlabel('x')
plt.ylabel('y')
# plt.show()


#2) Scatter plot at least 100 points of y=sin(3x) from x=-5 to x=5 in radians;
#label the axes as before and title the graph 'Sine Plot'
x=np.arange(-5,5,0.1)
y=np.sin(3*x)
plt.figure()
plt.title('Sine Plot')
plt.plot(x,y,'o')
plt.xlabel('x')
plt.ylabel('y')
# plt.show()


#3) Plot y=sin(x) in red and y=cos(x) in blue on one graph; label the axes as
#before, title the graph 'Trig Plot', and include a legend; as before, x is
#from -5 to 5
x=np.arange(-5,5,0.01)
y1 = np.sin(x)
y2 = np.cos(x)
plt.figure()
plt.title('Trig Plot')
plt.plot(x,y1,'r',label='sin')
plt.plot(x,y2,'b',label='cos')
plt.legend()
# plt.show()


#4) Create 2 side by side subplots; on the left show y=2x, and on the right show
#y=x**2; have them share the y-axis, and title the left graph "Line Plot" and
#the right graph "Parabola Plot"; also place a single scatter point at x=0
x=np.arange(-5,5,0.01)
fig, (ax1, ax2) = plt.subplots(1, 2,sharey=True)
ax1.plot(0, 0, 'o', color='red')
ax2.plot(0, 0, 'o', color='red')
ax1.plot(x,2*x)
ax2.plot(x, x**2)
ax1.set_title('Line Plot')
ax2.set_title('Parabola Plot')
plt.show()

#5) Plot y=x**2 from x=-5 to 5 again on a new plot, but this time set the x-
#limit from -3 to 3 and the y-limit from 1 to 10; label axes and title it
#"Trimmed Parabola"

x=np.arange(-5,5,0.01)
ax = plt.gca()
ax.plot(0, 0, 'o', color='red')
ax.set_xlim(-3,3)
ax.set_ylim(1,10)
ax.plot(x, x**2)
plt.xlabel('x')
plt.ylabel('y')
plt.suptitle('Trimmed Parabola')
plt.show()

#6) Create 2 subplots one on top of the other; plot y=2*x from x=-5 to 5, but
#with a log-scaled y axis on the top graph; on the bottom, plot y=log10(2*x) on
#a regularly scaled graph, i.e. not log-scaled; label the axes, have them share
#an x-axis, and title the top graph "Logarithmic Comparison" (don't worry
#about the bottom one)
x=np.arange(-5,5,0.01)
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax2.set_yscale('log')
ax1.set_title('Logarithmic Comparison')
ax1.plot(x, np.log10(2*x))
ax2.plot(x,2*x)
plt.show()

#7) Plot y=sin(x)+5 (red, labeled alpha) and y=sin(x) (blue, labeled beta) on
#one plot with labeled axes, a legend, titled 'Biased vs. Unbiased Sine
#Function', with the text in LateX format

x=np.arange(-5,5,0.01)
y1 = np.sin(x) + 5
y2 = np.sin(x)
plt.figure()
plt.title(r'$Biased$ $vs.$ $Unbiased$ $Sine$ $Function$')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x,y1,'r',label=r'$\alpha$')
plt.plot(x,y2,'b',label=r'$\beta$')
plt.legend()


#8) Return the font to normal and use this same graph again, but with the biased
#sine function using triangular markers, a marker size of 10, and an alpha of
#0.25, while the unbiased function uses dashed line style, a line width of 2.5,
#and an alpha of 0.75; label the respective functions as 'Biased' and 'Unbiased'

x=np.arange(-5,5,0.01)
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
y1 = np.sin(x) + 5
y2 = np.sin(x)
ax1.set_title('Biased')
ax2.set_title('Unbiased')
ax1.plot(x,y1,'r',marker='^',alpha=0.25, ms=10,label='biased')
ax2.plot(x,y2,'b',lw=2.5,alpha=0.75,ls='--',label='unbiased')
plt.legend()
plt.show()



#9) Create a bar graph with 5 categories marked with ticks 'A', 'B', 'C', 'D',
#and 'E'; the respective values are 1, 4, 5, 3, 2; label the x-axis 'categories',
#the y-axis 'values', and title the graph 'Bar Graph Demonstration'
plt.figure()
plt.xticks([1,2,3,4,5],['A','B','C','D','E'])
plt.bar([1,2,3,4,5],[1,4,5,3,2])
plt.title('Bar Graph Demonstration')
plt.xlabel('categories')
plt.ylabel('values')
plt.show()

#10) Generate a 10,000 element long array of random integers ranging from
#0 to 255, then create a subplot figure of a histogram (top) and the
#corresponding cumulative histogram (bottom) of the array; use 256 bins for
#both histograms, label the shared x-axis for both "pixel value", y-axis for
#the first histogram "pixel amount", y-axis for the cumulative histogram
#"cumulative amount", and title the top graph "Histogram Comparison"
plt.figure()
arr = np.random.randint(0,255,size=10000)
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.hist(arr,bins=256)
ax2.hist(arr,bins=256,cumulative=True)
ax1.set_xlabel('pixel value')
ax1.set_ylabel('pixel amount')
ax2.set_ylabel('cumulative amount')
# fig.set_title()
plt.suptitle('Histogram Comparison')
plt.show()