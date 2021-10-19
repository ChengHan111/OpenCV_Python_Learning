from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.optimize as opt
from scipy.interpolate import interp1d
import scipy.signal as sig
import scipy.linalg as lng

#1) Generate the function y=0.05*x**2+x+5 over x=[-5,5], and add Student's T noise
#with 5 degrees of freedom (df) to the function; then, use linear regression to
#fit a line with slope and intercept to the result; plot the result in a graph
#with the title showing the slope and intercept marked as such, x axis labeled
#'x', y axis labeled 'y', and showing the noisy data plotted using scatter points
#and the estimated line plotted as a line (may need to change color of line
#or points to see it clearly, or point size, feel free to alter such parameters
#as needed)
x = np.linspace(-5,5,100)
y = 0.05*x**2 + x + 5
yn = y + stats.norm.rvs(size=len(x))
res=stats.linregress(x,yn)
moment=stats.norm.moment(1)
plt.figure()
plt.scatter(x,yn)
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x,res.slope*x+res.intercept)
plt.show()


#2) Fit the (x,y) points (1,7),(6,6),(5,-4) to a parabola, that is, in the
#form of a*x**2+b*x+c using curve_fit; scatter plot the points on a graph
#and plot the resulting parabola as a line; label the x and y axes as before
#and title the graph with the marked coefficients of the parabola; then,
#use the root or fsolve optimization function to find the two roots of the 
#parabola and print the result

poly3=lambda x,a,b,c: a*x**2+b*x+c
arr=np.array([[1,7],[6,6],[5,-4]])
nvars,_=opt.curve_fit(poly3,arr[:,0],arr[:,1])
x2=np.linspace(-5,8,1000)
plt.figure()
plt.scatter(arr[:,0],arr[:,1])
plt.plot(x2,poly3(x2,nvars[0],nvars[1],nvars[2]))
plt.xlabel('x')
plt.ylabel('y')
plt.show()

p_res=opt.root(poly3,[0,6],args=(nvars[0], nvars[1], nvars[2]))
solved=opt.fsolve(poly3,[0,6],args=(nvars[0], nvars[1], nvars[2]))

print('p_res:', p_res)
print('solved:',solved)

#3) Find the minimum and maximum of sin(2x)+cos(x/2) within the range x=[0,7],
#then plot the function as a line and scatter plot the minimum and maximum
#points; label the axes as before and title the plot with the x values of 
#the minimum and maximum value locations

plt.figure()
function = lambda x: np.sin(2*x)+np.cos(x/2)
g_res=opt.differential_evolution(function, [(0,7)])
npar = lambda x: -function(x)
max_res=opt.differential_evolution(npar, [(0,7)])
x2 = np.linspace(0,7,1000)
plt.plot(x2, function(x2))
plt.scatter(g_res["x"],function(g_res["x"]))
plt.scatter(max_res["x"],function(max_res["x"]))
plt.xlabel('x')
plt.ylabel('y')
plt.title('Max_Value: %5.2f, Min_value: %5.2f' % (function(max_res["x"]), function(g_res["x"])))
plt.show()

#4) For parameters a=3, b=2, and c=1, calculate the values of a parabola
#ONLY at integers between -5 and 5 (inclusive). Then, use interpolation to
#determine the values of the parabola halfway between each integer (-4.5, 
#-3.5, and so on). Finally, scatter plot the parabola at the integers and
#then separately scatter plot the interpolated points; label them as such
#on a legend, label the axes accordingly, and title the plot "Interpolated
#Parabola"

plt.figure()
x = np.linspace(-5,5,100)
y = 3*x**2 + 2*x + 1
x2 = np.array([ _ for _ in np.arange(-4.5, 5.0, 1)])
parabola_interpolation = interp1d(x,y)
plt.scatter(x,y, label='origin points')
plt.scatter(x2,parabola_interpolation(x2), label='interpolated points')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.suptitle('Interpolated Parabola')

#5) Use the DFT matrix from the linalg module to determine and plot the
#Fourier Transform of a Rectangle Function

def rect(x):
    r=np.empty(len(x))
    for i in np.arange(0,len(x),1):
        if abs(x[i])>0.5:
            r[i]=0
        else:
            r[i]=1
    return r

s=0.01 #spacing
x4=np.arange(-5,5,s)
rect=rect(x4)
D=lng.dft(len(x4)) #Discrete Fourier Transform matrix
shift=lambda x: np.append(x[int(len(x)/2):],x[:int(len(x)/2)]) #Switch halves of vector
rect_shift=shift(rect) #shift vector before FT
Frect_s=np.dot(D,rect_shift) #Fourier Transform by multiplying DFT mat. with vector
Ft=shift(Frect_s)*s #shift resulting vector, then scale by x spacing
nu=x4/(len(x4)*s**2) #frequency=x/(len(x)*spacing**2)
plt.figure()
plt.plot(nu,Ft)
plt.show()